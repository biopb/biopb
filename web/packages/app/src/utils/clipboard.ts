/** Put *text* on the clipboard; whether it landed.
 *
 * The async API exists only in a secure context, so on a dashboard served over
 * plain http from another host `navigator.clipboard` is undefined, and a denied
 * permission rejects. Both fall back to a hidden selection and `execCommand`,
 * which still works there. A caller shows the failure, since `false` means the
 * text is not on the clipboard. */
export async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    /* fall through to the selection route */
  }
  try {
    const box = document.createElement("textarea");
    box.value = text;
    box.setAttribute("readonly", "");
    box.style.position = "fixed";
    box.style.opacity = "0";
    document.body.appendChild(box);
    box.select();
    const ok = document.execCommand("copy");
    box.remove();
    return ok;
  } catch {
    return false;
  }
}
