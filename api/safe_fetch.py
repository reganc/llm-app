"""Fetch a page the user asked to watch, without becoming a way into the network.

The address is resolved once, every result must be a public address, and the
connection is made to that address (so DNS can't change in between). Redirects
are followed by hand, only within the approved site, with the same checks.
"""
from __future__ import annotations

import asyncio
import ipaddress
import socket
import urllib.parse
from dataclasses import dataclass

import httpx

_MAX_BYTES = 3_000_000
_MAX_REDIRECTS = 3
_TIMEOUT_S = 20.0
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; llm-app page watch)",
    "Accept": "text/html,application/xhtml+xml,application/json,application/xml;q=0.9,text/plain;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
}
_TYPES = ("text/", "application/json", "application/xml", "application/xhtml", "application/rss",
          "application/atom", "application/ld+json")


class FetchError(Exception):
    """Safe to show to the user."""


@dataclass(frozen=True)
class Page:
    url: str
    content_type: str
    text: str


def _bare(host: str) -> str:
    return host.lower().rstrip(".").removeprefix("www.")


def host_of(url: str) -> str:
    """The site part of a URL the user may approve. Raises FetchError."""
    try:
        parts = urllib.parse.urlsplit((url or "").strip())
        port = parts.port
    except ValueError:
        raise FetchError("That is not a web address.") from None
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise FetchError("Only http and https addresses can be watched.")
    if parts.username or parts.password:
        raise FetchError("Addresses with a username or password can't be watched.")
    if port not in (None, 80, 443):
        raise FetchError("Only the standard web ports can be watched.")
    return parts.hostname.lower().rstrip(".")


def is_public(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address.split("%")[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped or ip.sixtofour or ip.teredo:
            return False
        if ip in ipaddress.ip_network("64:ff9b::/96"):  # NAT64 wraps an IPv4 address
            return False
    return ip.is_global and not ip.is_multicast and not ip.is_reserved


async def _lookup(host: str, port: int) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(str(info[4][0]) for info in infos))


async def _public_address(host: str, port: int) -> str:
    try:
        found = await asyncio.wait_for(_lookup(host, port), timeout=8)
    except (OSError, UnicodeError, asyncio.TimeoutError):
        raise FetchError("That site could not be found.") from None
    if not found or not all(is_public(a) for a in found):
        raise FetchError("That address is on a private network, which can't be watched.")
    return next((a for a in found if ":" not in a), found[0])


def _client() -> httpx.AsyncClient:
    # trust_env=False: a proxy from the environment would do its own DNS lookup.
    return httpx.AsyncClient(timeout=_TIMEOUT_S, follow_redirects=False, trust_env=False)


async def _get(client: httpx.AsyncClient, url: str, host: str) -> tuple[httpx.Response, bytes]:
    parts = urllib.parse.urlsplit(url)
    port = parts.port or (443 if parts.scheme == "https" else 80)
    address = await _public_address(host, port)
    literal = f"[{address}]" if ":" in address else address
    target = urllib.parse.urlunsplit((parts.scheme, literal, parts.path or "/", parts.query, ""))
    body = b""
    async with client.stream("GET", target, headers={**_HEADERS, "Host": host},
                             extensions={"sni_hostname": host}) as response:
        if response.status_code == 200:
            kind = response.headers.get("content-type", "").lower()
            if kind and not kind.startswith(_TYPES):
                raise FetchError("That address is not a text page.")
            async for chunk in response.aiter_bytes():
                body += chunk
                if len(body) > _MAX_BYTES:
                    raise FetchError("That page is too large to watch.")
    return response, body


async def fetch(url: str, approved_host: str) -> Page:
    async with _client() as client:
        try:
            return await asyncio.wait_for(_follow(client, url, approved_host), timeout=60)
        except (httpx.HTTPError, httpx.InvalidURL) as e:
            raise FetchError(f"The page could not be loaded ({type(e).__name__}).") from None
        except asyncio.TimeoutError:
            raise FetchError("The page took too long to load.") from None


async def _follow(client: httpx.AsyncClient, url: str, approved_host: str) -> Page:
    for _ in range(_MAX_REDIRECTS + 1):
        host = host_of(url)
        if _bare(host) != _bare(approved_host):
            raise FetchError(f"The page redirects to another site ({host}), which you have not approved.")
        response, body = await _get(client, url, host)
        if response.status_code in (301, 302, 303, 307, 308) and response.headers.get("location"):
            url = urllib.parse.urljoin(url, response.headers["location"])
            continue
        if response.status_code != 200:
            raise FetchError(f"The site answered with an error ({response.status_code}).")
        encoding = response.charset_encoding or "utf-8"
        try:
            text = body.decode(encoding, errors="replace")
        except LookupError:
            text = body.decode("utf-8", errors="replace")
        return Page(url, response.headers.get("content-type", "").lower(), text)
    raise FetchError("The page redirects too many times.")
