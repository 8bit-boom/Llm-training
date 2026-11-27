"""Hugging Face refusals (401 / 403, a gated or private model, a token that is missing or not valid) as one plain sentence that points to the app's Settings card.

The token reaches the worker only as the environment variable HF_TOKEN (the program sets it for every job); `huggingface_hub` reads it itself. Nothing here ever
puts the token into a text: `message` only says whether one was sent."""
import os
import re

# what huggingface_hub / transformers / datasets / urllib say; each needs a sign that the answer came from the Hub (an exception name, or the Hub's address next to the status)
_NAMES = re.compile(r"\b(GatedRepoError|GatedDatasetError)\b")
_STATUS = re.compile(r"(\b40[13]\s+(Client Error|Forbidden|Unauthorized)|HTTP Error 40[13]|status(?:_code)?[ =:]+40[13]\b|\b40[13] Client Error)", re.I)
_HUB = re.compile(r"(huggingface\.co|hf\.co|hf-mirror|HF_ENDPOINT|Hugging Face)", re.I)
_PHRASES = re.compile(r"(cannot access gated repo|access to model \S+ is restricted|you are trying to access a gated repo|you must have access to it and be authenticated"
                      r"|restricted and you are not in the authorized list|invalid user token|invalid credentials in authorization header"
                      r"|make sure you are authenticated|repository not found for url)", re.I)

SETTINGS_WORDS = "the 'Hugging Face token' card on the Training setup page (Settings)"


def is_access_error(exc_or_text) -> bool:
    """True when the failure is Hugging Face refusing a download (401 / 403 / gated / private / bad token). A 404 for a misspelled name is NOT one."""
    if isinstance(exc_or_text, BaseException):
        if any(c.__name__ in ("GatedRepoError", "GatedDatasetError") for c in type(exc_or_text).__mro__):
            return True
        resp = getattr(exc_or_text, "response", None)
        code = getattr(resp, "status_code", None)
        if code in (401, 403):
            return True
        text = f"{type(exc_or_text).__name__}: {exc_or_text}"
        cause = getattr(exc_or_text, "__cause__", None)
        if cause is not None and cause is not exc_or_text and is_access_error(cause):
            return True
    else:
        text = str(exc_or_text or "")
    if _NAMES.search(text) or _PHRASES.search(text):
        return True
    return bool(_STATUS.search(text) and _HUB.search(text))


def token_sent() -> bool:
    return bool((os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip())


def message(subject: str = "") -> str:
    """The plain sentence for the page. `subject` is the model name when it is known."""
    what = f"the model {subject}" if subject else "the model"
    if token_sent():
        return (f"Hugging Face refused the download of {what}. The Hugging Face token this program sent was not accepted for it: either the token is not valid any more, "
                "or this model is gated and your account has not been given access yet (open the model's page on huggingface.co, accept its terms, and wait for the approval). "
                f"Check the token in {SETTINGS_WORDS} and try again.")
    return (f"Hugging Face refused the download of {what}. The model is gated or private, or a Hugging Face token is needed and none is saved. "
            f"Make a token at huggingface.co/settings/tokens (read access is enough), save it in {SETTINGS_WORDS}, accept the model's terms on its page if it has any, and try again.")
