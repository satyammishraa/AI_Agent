import streamlit as st
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
import json

# -----------------------
# CONFIG
# -----------------------
MODEL_NAME = "llama-3.2-70b-versatile"  # use your model
BASE_URL = "https://api.groq.com/openai/v1"

st.set_page_config(page_title="AI Use Case Generator", page_icon="🤖")

# secure retrieval of API key from Streamlit secrets
api_key = st.secrets.get("GROQ_API_KEY")
if not api_key:
    st.error("❌ GROQ_API_KEY is missing. Add it to .streamlit/secrets.toml or Streamlit's Secrets.")
    st.stop()

HEADERS = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}


# -----------------------
# Helper: call Groq chat completions (OpenAI-compatible endpoint)
# -----------------------
def extract_content(resp_json: dict) -> str:
    """Robustly extract assistant text from Groq/OpenAI-compatible response."""
    if not isinstance(resp_json, dict):
        return json.dumps(resp_json)
    choices = resp_json.get("choices") or []
    if choices:
        first = choices[0]
        # OpenAI-compatible chat completion shape:
        if isinstance(first, dict):
            msg = first.get("message")
            if isinstance(msg, dict) and "content" in msg:
                return msg["content"]
            # fallback to text field (old style)
            if "text" in first:
                return first["text"]
            # sometimes message may be a string
            if isinstance(first.get("message"), str):
                return first.get("message")
    # common alternatives
    for k in ("output_text", "response", "text"):
        val = resp_json.get(k)
        if isinstance(val, str):
            return val
    # last resort: stringify whole response
    return json.dumps(resp_json)


def groq_chat_completion(
    messages: List[Dict], model: str = MODEL_NAME, max_tokens: int = 512, temperature: float = 0.0
) -> Optional[str]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        resp = requests.post(f"{BASE_URL}/chat/completions", headers=HEADERS, json=payload, timeout=30)
    except requests.RequestException as e:
        st.error(f"Network error when contacting Groq API: {e}")
        return None

    if resp.status_code != 200:
        # Show short error to user; logs contain full text if needed
        st.error(f"Groq API returned HTTP {resp.status_code}: {resp.text[:200]}")
        return None

    try:
        data = resp.json()
    except ValueError:
        st.error("Unable to decode response from Groq API (invalid JSON).")
        return None

    return extract_content(data)


def clean_list_lines(text: str) -> List[str]:
    if not text:
        return []
    lines = []
    for l in text.splitlines():
        l = l.strip()
        if not l:
            continue
        # drop leading bullets/numbers
        l = l.lstrip("-•*0123456789. )")
        lines.append(l.strip())
    return lines


# -----------------------
# WEB SCRAPER
# -----------------------
class WebBrowserTools:
    def __init__(self, model: str = MODEL_NAME):
        self.model = model

    def scrape_company_info(self, company_name: str) -> Dict:
        description = None
        try:
            url = f"https://en.wikipedia.org/wiki/{company_name.replace(' ', '_')}"
            resp = requests.get(url, timeout=8)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                para = soup.find("p")
                if para and para.get_text(strip=True):
                    description = para.get_text().strip()
        except Exception:
            description = None

        if not description:
            description = self.generate_description_with_groq(company_name) or f"{company_name} - description not found."

        offerings = self.generate_offerings(description)
        focus = self.generate_focus_areas(description)

        return {
            "company": company_name,
            "description": description,
            "offerings": offerings,
            "focus": focus,
        }

    def generate_description_with_groq(self, company_name: str) -> Optional[str]:
        prompt = f"Write a 2-3 line professional description about {company_name}."
        messages = [{"role": "user", "content": prompt}]
        return groq_chat_completion(messages, model=self.model, max_tokens=200, temperature=0.0)

    def generate_offerings(self, description: str) -> List[str]:
        prompt = f"From this description, list the main offerings/products/services (one per line):\n\n{description}"
        messages = [{"role": "user", "content": prompt}]
        out = groq_chat_completion(messages, model=self.model, max_tokens=250, temperature=0.0) or ""
        return clean_list_lines(out)

    def generate_focus_areas(self, description: str) -> List[str]:
        prompt = f"From this description, what strategic focus areas does the company have? (e.g., AI, Cloud, Healthcare). List one per line."
        messages = [{"role": "user", "content": prompt}]
        out = groq_chat_completion(messages, model=self.model, max_tokens=200, temperature=0.0) or ""
        return clean_list_lines(out)


# -----------------------
# AGENTS & ORCHESTRATOR
# -----------------------
class ResearchAgent:
    def __init__(self):
        self.browser_tools = WebBrowserTools()

    def research_company(self, company_name: str) -> Dict:
        return self.browser_tools.scrape_company_info(company_name)


class MarketAnalysisAgent:
    def generate_use_cases(self, company_info: Dict) -> List[str]:
        description = company_info.get("description", "")
        prompt = f"Suggest 5 AI/GenAI/ML use cases for the following company.\n\n{description}\n\nList each use case on a new line with a 1-2 sentence explanation."
        messages = [{"role": "user", "content": prompt}]
        out = groq_chat_completion(messages, model=MODEL_NAME, max_tokens=600, temperature=0.2) or ""
        return clean_list_lines(out)


class MultiAgentSystem:
    def __init__(self):
        self.research_agent = ResearchAgent()
        self.market_agent = MarketAnalysisAgent()

    def run(self, company_name: str) -> Dict:
        company_info = self.research_agent.research_company(company_name)
        use_cases = self.market_agent.generate_use_cases(company_info)
        return {"company_info": company_info, "use_cases": use_cases}


# -----------------------
# STREAMLIT UI
# -----------------------
st.title("🤖 Multi-Agent AI Use Case Generator")
company_name = st.text_input("Enter a company name to generate AI/ML/GenAI use cases and resources:")

if company_name:
    orchestrator = MultiAgentSystem()
    with st.spinner("Agents are running..."):
        results = orchestrator.run(company_name)

    if results:
        st.success("✅ Completed!")
        info = results["company_info"]

        st.subheader("📄 Company Information")
        st.write(f"**Company:** {info.get('company')}")
        st.write(f"**Description:** {info.get('description')}")
        st.write("**Offerings:**")
        st.write(info.get("offerings", []))
        st.write("**Strategic Focus Areas:**")
        st.write(info.get("focus", []))

        st.subheader("🚀 AI/GenAI Use Cases")
        for case in results.get("use_cases", []):
            st.write("-", case)
    else:
        st.error("No results returned. Check the Groq API key and logs for more details.")







