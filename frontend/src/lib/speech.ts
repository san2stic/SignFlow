export function speak(text: string): void {
  if (!("speechSynthesis" in window) || !text.trim()) {
    return;
  }
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "fr-FR";
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
}
