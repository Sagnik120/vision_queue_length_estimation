import os
import time
from dotenv import load_dotenv

load_dotenv()

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.output_parser import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


SYSTEM_PROMPT = """You are a real-time crowd management AI assistant deployed at a public venue.
Generate concise, actionable advisories for venue staff from live queue metrics.
Rules: Under 2 sentences. Specific numbers. One clear action. No "I" or "the system".
Queue < 5: informational tone. Queue 5-10: advisory tone. Queue > 10: urgent tone."""

HUMAN_PROMPT = """Queue metrics:
- Queue size: {queue_size} people
- Queue length: ~{queue_length_m:.1f} meters
- Scene: {scene_label} ({scene_confidence:.0%} confidence)
- Location: {location}
- Time: {timestamp}
Generate advisory:"""


class QueueAdvisoryAgent:
    """
    LangChain + Gemini advisory agent.

    KEY FIX from previous version: fallback advisory is now always computed
    fresh from the current queue_size. Previously the whole get_advisory()
    call was rate-limited, so the first "0 people" message was cached and
    returned for 14 consecutive frames even as queue_size grew to 5-8.
    Now only the LLM call is rate-limited. Fallback always reflects reality.
    """

    MIN_INTERVAL_SECONDS = 60

    def __init__(self, location="Main Entrance", pixels_per_meter=90.0):
        self.location         = location
        self.pixels_per_meter = pixels_per_meter
        self._last_llm_time   = 0.0
        self._chain           = None
        self._available       = False
        self._quota_failed    = False

        if os.getenv("DISABLE_GEMINI", "0") == "1":
            print("Advisory agent: rule-based fallback mode.")
            return

        if not LANGCHAIN_AVAILABLE:
            print("Advisory agent: langchain-google-genai not installed. Using fallback.")
            return

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("Advisory agent: no API key. Using rule-based fallback.")
            return

        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-lite",
                google_api_key=api_key,
                temperature=0.3,
                max_output_tokens=80,
                max_retries=1,
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human",  HUMAN_PROMPT),
            ])
            self._chain     = prompt | llm | StrOutputParser()
            self._available = True
            print("Advisory agent ready (Gemini free tier).")
        except Exception as e:
            print(f"Advisory agent init failed ({e}). Using fallback.")

    def get_advisory(self, queue_size, queue_length_px,
                     scene_label, scene_confidence):
        queue_length_m = queue_length_px / self.pixels_per_meter
        timestamp      = time.strftime("%H:%M:%S")
        now            = time.time()

        if (self._available and self._chain is not None
                and not self._quota_failed
                and now - self._last_llm_time >= self.MIN_INTERVAL_SECONDS):
            self._last_llm_time = now
            try:
                advisory = self._chain.invoke({
                    "queue_size":       queue_size,
                    "queue_length_m":   queue_length_m,
                    "scene_label":      scene_label,
                    "scene_confidence": scene_confidence,
                    "location":         self.location,
                    "timestamp":        timestamp,
                })
                return advisory.strip().replace("\n", " ")
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    self._quota_failed = True
                    print("Advisory agent: quota exhausted. Using fallback.")

        # fallback always uses live queue_size — never stale
        return self._fallback_advisory(queue_size, queue_length_m, scene_label)

    def _fallback_advisory(self, queue_size, queue_length_m, scene_label):
        if scene_label not in ("queue",):
            return f"Scene: {scene_label}. Monitoring continues."
        if queue_size >= 12:
            return (f"URGENT: {queue_size} people queuing over "
                    f"{queue_length_m:.1f}m. Open additional counter immediately.")
        if queue_size >= 6:
            return (f"ADVISORY: Queue of {queue_size} people "
                    f"({queue_length_m:.1f}m). Consider opening additional counter.")
        if queue_size >= 3:
            return (f"INFO: Queue of {queue_size} people "
                    f"({queue_length_m:.1f}m). No action needed.")
        return "INFO: No significant queue detected."
