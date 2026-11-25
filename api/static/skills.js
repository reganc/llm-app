// Skills: saved instructions, used with /name, in routines and from Telegram.

export function initSkills({ $, el, api, toast }) {
  const call = {
    list: () => api.json("/v1/skills"),
    create: (b) => api.json("/v1/skills", { method: "POST", body: JSON.stringify(b) }),
    update: (n, b) => api.json(`/v1/skills/${encodeURIComponent(n)}`, { method: "PUT", body: JSON.stringify(b) }),
    remove: (n) => api.json(`/v1/skills/${encodeURIComponent(n)}`, { method: "DELETE" }),
  };
  const modal = $("#skillsModal"), list = $("#skillsList"), form = $("#skillForm"), add = $("#skillNewBtn");
  let skills = [];
  let editing = null;

  const detail = (e) => {
    try {
      const d = JSON.parse(e.message.slice(e.message.indexOf("{"))).detail;
      return typeof d === "string" ? d : d?.[0]?.msg?.replace(/^Value error, /, "") || e.message;
    } catch { return e.message; }
  };

  async function refresh() {
    skills = (await call.list()).skills;
    return skills;
  }

  function showForm(skill) {
    editing = skill?.name || null;
    form.hidden = false;
    add.hidden = list.hidden = true;
    for (const key of ["name", "description", "instructions"]) form.elements[key].value = skill?.[key] || "";
    form.elements.name.disabled = !!editing;
    (editing ? form.elements.instructions : form.elements.name).focus();
  }

  function showList() {
    form.hidden = true;
    add.hidden = list.hidden = false;
    list.replaceChildren(...skills.map(s => el("div", { class: "skill-row" }, [
      el("div", { class: "skill-text" }, [
        el("div", {}, [el("code", { text: `/${s.name}` }), el("span", { class: "muted-small", text: s.description ? ` ${s.description}` : "" })]),
        el("p", { class: "muted-small skill-body", text: s.instructions }),
      ]),
      el("button", { type: "button", class: "ghost-btn", text: "Edit", onclick: () => showForm(s) }),
      el("button", { type: "button", class: "ghost-btn danger", text: "Delete", onclick: async () => {
        if (!confirm(`Delete the skill /${s.name}?`)) return;
        try { await call.remove(s.name); await refresh(); showList(); }
        catch (e) { toast(detail(e), "error", 5000); }
      } }),
    ])));
    if (!skills.length) list.append(el("p", { class: "muted", text: "No skills yet." }));
  }

  form.onsubmit = async (e) => {
    e.preventDefault();
    const body = {
      name: editing || form.elements.name.value.trim(),
      description: form.elements.description.value.trim(),
      instructions: form.elements.instructions.value.trim(),
    };
    const save = form.querySelector("[type=submit]");
    save.disabled = true;
    try {
      await (editing ? call.update(editing, body) : call.create(body));
      toast("Skill saved", "success", 1500);
      await refresh();
      showList();
    } catch (err) { toast(detail(err), "error", 5000); }
    save.disabled = false;
  };
  form.querySelector(".cancel").onclick = showList;
  add.onclick = () => showForm(null);

  async function open() {
    try { await refresh(); } catch (e) { toast(`Skills unavailable: ${e.message}`, "error", 5000); return; }
    showList();
    modal.showModal();
  }
  $("#skillsBtn").onclick = open;
  return { open, refresh, list: () => skills };
}
