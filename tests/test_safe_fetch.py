"""Watched pages can't be used to reach private addresses."""
from __future__ import annotations

import asyncio

import httpx
import pytest

import safe_fetch as sf

HTML = {"content-type": "text/html; charset=utf-8"}


def run(c):
    return asyncio.run(c)


@pytest.fixture
def net(monkeypatch):
    """net.dns host -> addresses; net.pages (host, path) -> Response."""
    class Net:
        dns = {"shop.example": ["93.184.216.34"], "www.shop.example": ["93.184.216.34"],
               "other.example": ["93.184.216.35"]}
        pages, seen = {}, []
    n = Net()
    n.pages, n.seen, n.dns = {}, [], dict(Net.dns)

    async def lookup(host, _port):
        if host not in n.dns:
            raise OSError("no such host")
        found = n.dns[host]
        return found.pop(0) if found and isinstance(found[0], list) else found

    def handle(request):
        n.seen.append((request.url.host, request.headers["host"], request.url.path,
                       request.extensions.get("sni_hostname")))
        r = n.pages.get((request.headers["host"], request.url.path))
        if isinstance(r, Exception):
            raise r
        return r or httpx.Response(200, headers=HTML, text="hello")
    monkeypatch.setattr(sf, "_lookup", lookup)
    monkeypatch.setattr(sf, "_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    return n


@pytest.mark.parametrize("url", [
    "ftp://shop.example/", "file:///etc/passwd", "javascript:alert(1)", "shop.example", "", "http://",
    "http://user:pw@shop.example/", "http://shop.example:8030/", "http://shop.example:22/", "http://[::1",
    "http://shop.example:99999/", "gopher://shop.example/",
])
def test_unsuitable_addresses(url):
    with pytest.raises(sf.FetchError):
        sf.host_of(url)


@pytest.mark.parametrize("address", [
    "127.0.0.1", "10.0.0.5", "172.17.0.1", "192.168.1.1", "169.254.169.254", "100.64.0.1", "0.0.0.0",
    "224.0.0.1", "255.255.255.255", "240.0.0.1", "::1", "::", "fe80::1", "fe80::1%eth0", "fc00::1",
    "fd7a:115c:a1e0::1", "::ffff:127.0.0.1", "::ffff:10.0.0.1", "64:ff9b::7f00:1", "2002:7f00:1::",
    "2001::1", "ff02::1", "192.0.2.1", "198.18.0.1", "not an address",
])
def test_private_addresses(address):
    assert sf.is_public(address) is False


@pytest.mark.parametrize("address", ["93.184.216.34", "8.8.8.8", "2606:4700:4700::1111"])
def test_public_addresses(address):
    assert sf.is_public(address) is True


def test_connects_to_the_checked_address_with_the_right_name(net):
    page = run(sf.fetch("https://shop.example/p?q=1#frag", "shop.example"))
    assert page.text == "hello"
    assert net.seen == [("93.184.216.34", "shop.example", "/p", "shop.example")]


@pytest.mark.parametrize("found", [["10.0.0.1"], ["93.184.216.34", "127.0.0.1"], ["::1"], []])
def test_any_private_answer_is_refused_without_connecting(net, found):
    net.dns["shop.example"] = found
    with pytest.raises(sf.FetchError, match="private"):
        run(sf.fetch("https://shop.example/", "shop.example"))
    assert net.seen == []


def test_the_name_is_looked_up_once_per_request_so_dns_cannot_change(net):
    calls = []
    inner = sf._lookup

    async def counted(host, port):
        calls.append(host)
        return await inner(host, port)
    sf._lookup = counted
    try:
        run(sf.fetch("https://shop.example/", "shop.example"))
    finally:
        sf._lookup = inner
    assert calls == ["shop.example"]


def test_literal_private_ip_is_refused(net):
    net.dns["127.0.0.1"] = ["127.0.0.1"]
    with pytest.raises(sf.FetchError):
        run(sf.fetch("http://127.0.0.1/", "127.0.0.1"))


def test_redirects_stay_on_the_approved_site(net):
    net.pages[("shop.example", "/")] = httpx.Response(301, headers={"location": "https://www.shop.example/new"})
    assert run(sf.fetch("https://shop.example/", "shop.example")).url == "https://www.shop.example/new"
    net.pages[("shop.example", "/")] = httpx.Response(302, headers={"location": "https://other.example/"})
    with pytest.raises(sf.FetchError, match="other.example"):
        run(sf.fetch("https://shop.example/", "shop.example"))


@pytest.mark.parametrize("to", ["http://127.0.0.1/", "http://169.254.169.254/latest", "file:///etc/passwd",
                                "http://shop.example:8030/", "//localhost/"])
def test_redirects_to_private_places_are_refused(net, to):
    net.pages[("shop.example", "/")] = httpx.Response(302, headers={"location": to})
    with pytest.raises(sf.FetchError):
        run(sf.fetch("https://shop.example/", "shop.example"))
    assert len(net.seen) == 1


def test_a_redirect_is_checked_again_even_on_the_same_site(net):
    net.dns["shop.example"] = [["93.184.216.34"], ["10.0.0.1"]]
    net.pages[("shop.example", "/")] = httpx.Response(302, headers={"location": "/b"})
    with pytest.raises(sf.FetchError, match="private"):
        run(sf.fetch("https://shop.example/", "shop.example"))


def test_redirect_loop(net):
    net.pages[("shop.example", "/")] = httpx.Response(302, headers={"location": "/"})
    with pytest.raises(sf.FetchError, match="too many"):
        run(sf.fetch("https://shop.example/", "shop.example"))


@pytest.mark.parametrize("reply,word", [
    (httpx.Response(403), "403"), (httpx.Response(500), "500"),
    (httpx.Response(200, headers={"content-type": "image/png"}, content=b"x"), "text page"),
    (httpx.Response(200, headers={"content-type": "application/octet-stream"}, content=b"x"), "text page"),
    (httpx.Response(200, headers=HTML, content=b"x" * (sf._MAX_BYTES + 1)), "too large"),
    (httpx.ConnectError("https://secret"), "ConnectError"), (httpx.ReadTimeout("x"), "ReadTimeout"),
])
def test_failures_are_explained(net, reply, word):
    net.pages[("shop.example", "/")] = reply
    with pytest.raises(sf.FetchError) as e:
        run(sf.fetch("https://shop.example/", "shop.example"))
    assert word in str(e.value) and "secret" not in str(e.value)


def test_unknown_site_and_charsets(net):
    with pytest.raises(sf.FetchError, match="found"):
        run(sf.fetch("https://nowhere.example/", "nowhere.example"))
    net.pages[("shop.example", "/")] = httpx.Response(200, headers={"content-type": "text/html; charset=nope"}, content="é".encode())
    assert run(sf.fetch("https://shop.example/", "shop.example")).text


def test_the_real_client_ignores_proxies_and_redirects():
    client = sf._client()
    assert client.follow_redirects is False and client.trust_env is False
