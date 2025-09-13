# app.py
import streamlit as st
import requests
import re
from typing import Dict, List
from groq import Groq

# --------------- Configuration ---------------
MODEL_NAME = "llama-3.1-70b-versatile"  # change if you want another Groq model
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# --------------- Helpers / Fallbacks ---------------
def _simple_offerings_from_description(description: str, max_items: int = 3) -> List[str]:
    """Heuristic fallback: split sentences and comma/and lists to produce 1-3 offerings."""
    if not description:
        return []
    # Split into sentences and fragments
    fragments = re.split(r'[.\n]', description)
    candidates = []
    for fragment in fragments:
        fragment = fragment.strip()
        if not fragment:
            continue
        parts = re.split(r',| and | & |;|-', fragment)
        for p in parts:
            p = p.strip()
            # ignore super short words
            if len(p) >= 10 and p.lower() not in (c.lower() for c in candidates):
                candidates.append(p)
            if len(candidates) >= max_items:
                break
        if len(candidates) >= max_items:
            break
    # If nothing found, return truncated description as single offering
    if not candidates:
        return [description.strip()[:140]]
    return candidates[:max_items]

def _simple_focus_from_description(description: str, max_items: int = 3) -> List[str]:
    """Heuristic fallback to produce 2-3 focus areas from description."""
    keywords = ["cloud", "ai", "ml", "mobile", "saas", "hardware", "biotech", "health", "finance", "e-commerce", "logistics", "security", "analytics"]
    found = []
    for kw in keywords:
        if kw in description.lower():
            # present a nice label
            label = {
                "ai": "AI / Machine Learning",
                "ml": "AI / Machine Learning",
                "cloud": "Cloud Infrastructure",
                "saas": "SaaS / Platform",
                "mobile": "Mobile & Apps",
                "hardware": "Hardware / Devices",
                "biotech": "Biotech / Life Sciences",
                "health": "Healthcare Solutions",
                "finance": "Financial Services",
                "e-commerce": "E-commerce / Retail",
                "logistics": "Logistics / Supply Chain",
                "security": "Security / Privacy",
                "analytics": "Data Analytics"
            }.get(kw, kw.title())
            if label not in found:
                found.append(label)
        if len(found) >= max_items:
            break
    if not found:
        # fallback to generic recommendations
        return ["Product/Market Fit", "Customer Experience", "Operational Efficiency"][:max_items]
    return found[:max_items]

def _simple_use_cases_from_industry(industry_data: Dict) -> str:
    """Fallback use-cases if Groq fails - template-driven based on industry."""
    industry = (industry_data.get("industry") or "General").lower()
    desc = industry_data.get("description", "")
    offerings = industry_data.get("offerings", [])
    focus = industry_data.get("strategic_focus", [])

    bullets = []
    if "automotiv" in industry or "vehicle" in industry:
        bullets = [
            "- Predictive maintenance using telemetry + ML to reduce downtime.",
            "- Visual inspection of parts using computer vision.",
            "- Drive-assist personlization (GenAI for driver prompts & UX)."
        ]
    elif "finance" in industry or "bank" in industry:
        bullets = [
            "- Fraud detection using anomaly detection models.",
            "- Smart financial summaries & compliance automation with GenAI.",
            "- Customer support & chatbot for common inquiries."
        ]
    elif "e-commerce" in industry or "retail" in industry:
        bullets = [
            "- Personalized product recommendations (recs + ranking).",
            "- Automated product description generation using GenAI.",
            "- Demand forecasting for inventory optimization."
        ]
    elif "health" in industry or "medical" in industry:
        bullets = [
            "- Medical imaging assistance using CV models (triage).",
            "- Clinical note summarization and coding with GenAI.",
            "- Patient triage chatbot and remote-monitoring analytics."
        ]
    elif "technology" in industry or "software" in industry:
        bullets = [
            "- Automated code/documentation generation with GenAI assistants.",
            "- Observability analytics (anomaly detection, root-cause).",
            "- Customer success automation and knowledge-base search."
        ]
    else:
        # general fallback
        bullets = [
            "- Improve customer experience using personalized GenAI chat assistants.",
            "- Automate reporting and internal knowledge search with semantic search + GenAI.",
            "- Operational efficiency via predictive analytics and automation."
        ]

    # Append a small tailored bullet referencing offerings/focus
    if offerings:
        bullets.append(f"- Tailored solution idea: adapt GenAI to {offerings[0]}.")
    if focus:
        bullets.append(f"- Strategic alignment: focus on {focus[0]} to unlock growth.")

    return "\n".join(bullets)

# --------------- Groq wrapper functions ---------------
class WebBrowserTools:
    def __init__(self, model: str = MODEL_NAME):
        self.model = model

    def scrape_company_info(self, company_name: str) -> Dict:
        """Scrape Wikipedia for summary; if missing or too generic, use LLM to generate description."""
        description = ""
        try:
            company_name_formatted = company_name.replace(" ", "_")
            wiki_resp = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{company_name_formatted}", timeout=8)
            if wiki_resp.status_code == 200:
                data = wiki_resp.json()
                description = data.get("extract", "") or ""
                # avoid disambiguation pages
                if "may refer to:" in description.lower() or len(description.strip()) < 40:
                    description = ""
        except Exception as e:
            # swallow and fallback to LLM
            print("Wiki fetch error:", e)
            description = ""

        if not description:
            description = self.generate_description_with_groq(company_name)

        offerings = self.generate_offerings(description)
        if not offerings:
            offerings = _simple_offerings_from_description(description)

        strategic_focus = self.generate_focus_areas(description)
        if not strategic_focus:
            strategic_focus = _simple_focus_from_description(description)

        return {
            "description": description,
            "products": offerings,
            "focus_areas": strategic_focus
        }

    def _call_chat_completion(self, prompt: str, system_role: str = "You are a helpful AI assistant.", max_tokens: int = 200):
        """
        Centralized chat call with the correct Groq param name.
        Returns str or raises exception.
        """
        # Important: Groq uses `max_completion_tokens` for chat completions.
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_completion_tokens=max_tokens
        )
        return response.choices[0].message.content

    def generate_description_with_groq(self, company_name: str) -> str:
        prompt = f"Write a concise 2-3 line professional description about the company '{company_name}'."
        try:
            text = self._call_chat_completion(prompt, system_role="You are a concise business copywriter.", max_tokens=180)
            return text.strip()
        except Exception as e:
            # log the error for debug
            st.session_state.setdefault("last_api_error", [])
            st.session_state["last_api_error"].append(str(e))
            print("Groq API Error (description):", e)
            return f"{company_name} (description unavailable due to API error)."

    def generate_focus_areas(self, description: str) -> List[str]:
        prompt = f"Suggest 2-3 strategic focus areas for the company based on this description:\n\n{description}\n\nReturn each focus area on a new line."
        try:
            text = self._call_chat_completion(prompt, system_role="You are a strategic business consultant.", max_tokens=120)
            items = [line.strip(" -•") for line in text.splitlines() if line.strip()]
            return items[:3]
        except Exception as e:
            st.session_state.setdefault("last_api_error", [])
            st.session_state["last_api_error"].append(str(e))
            print("Groq API Error (focus):", e)
            return []

    def generate_offerings(self, description: str) -> List[str]:
        prompt = f"Based on this company description, list 2-3 main products or services they offer. Return each on a new line:\n\n{description}"
        try:
            text = self._call_chat_completion(prompt, system_role="You are an expert product analyst.", max_tokens=120)
            items = [line.strip(" -•") for line in text.splitlines() if line.strip()]
            return items[:3]
        except Exception as e:
            st.session_state.setdefault("last_api_error", [])
            st.session_state["last_api_error"].append(str(e))
            print("Groq API Error (offerings):", e)
            return []

class BaseAgent:
    def __init__(self, name: str):
        self.name = name
        self.log = []

    def log_action(self, action: str):
        self.log.append(f"{self.name}: {action}")

class ResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__("Research Agent")
        self.browser_tools = WebBrowserTools()

    def research_company(self, company_name: str) -> Dict:
        self.log_action(f"Researching {company_name}")
        company_data = self.browser_tools.scrape_company_info(company_name)
        industry = self._classify_industry(company_data["description"])
        return {
            "company": company_name,
            "industry": industry,
            "offerings": company_data["products"],
            "strategic_focus": company_data["focus_areas"],
            "description": company_data["description"]
        }

    def _classify_industry(self, description: str) -> str:
        if not description:
            return "General Industry"
        desc = description.lower()
        if "automotive" in desc or "vehicle" in desc:
            return "Automotive"
        if "bank" in desc or "finance" in desc:
            return "Finance"
        if "e-commerce" in desc or "retail" in desc or "shop" in desc:
            return "E-commerce/Retail"
        if "software" in desc or "technology" in desc or "saas" in desc:
            return "Technology"
        if "entertainment" in desc or "media" in desc:
            return "Entertainment/Media"
        if "health" in desc or "medical" in desc or "clinic" in desc:
            return "Healthcare"
        return "General Industry"

class MarketAnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__("Market Analysis Agent")
        self.browser_tools = WebBrowserTools()

    def generate_use_cases(self, industry_data: Dict) -> str:
        self.log_action("Generating AI/ML/GenAI use cases")
        prompt = f"""You are an AI business consultant.
Analyze the following industry and suggest concise AI/ML and Generative AI (GenAI) use cases (return bullet points, each on its own line):

Industry: {industry_data['industry']}
Description: {industry_data['description']}
Offerings: {industry_data['offerings']}
Strategic Focus Areas: {industry_data['strategic_focus']}

Please suggest:
- Practical AI/ML/GenAI solutions
- Improvements to customer experience, operations, supply chain
- Internal GenAI solutions (chatbots, automated reporting, document search)
"""
        try:
            text = self.browser_tools._call_chat_completion(prompt, system_role="You are a consultant specializing in AI for business.", max_tokens=420)
            return text.strip()
        except Exception as e:
            st.session_state.setdefault("last_api_error", [])
            st.session_state["last_api_error"].append(str(e))
            print("Groq API Error (use cases):", e)
            # deterministic fallback
            return _simple_use_cases_from_industry(industry_data)

class MultiAgentSystem:
    def __init__(self):
        self.research_agent = ResearchAgent()
        self.market_agent = MarketAnalysisAgent()

    def run(self, company_name: str) -> Dict:
        company_info = self.research_agent.research_company(company_name)
        use_cases = self.market_agent.generate_use_cases(company_info)
        return {
            "company_info": company_info,
            "ai_use_cases": use_cases
        }

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Multi-Agent AI Use Case Generator", page_icon="🤖")
st.title("🤖 Multi-Agent AI Use Case Generator")
st.write("Enter a company name to generate AI/ML/GenAI use cases and resources:")

company_name = st.text_input("Company Name")

if st.button("Run Agents 🚀"):
    if not company_name:
        st.warning("Please enter a company name first!")
    else:
        orchestrator = MultiAgentSystem()
        with st.spinner("Agents are working..."):
            results = orchestrator.run(company_name)

        st.success("✅ Completed!")
        company_info = results["company_info"]
        use_cases = results["ai_use_cases"]

        st.header("📄 Company Information")
        st.markdown(f"**Company:** {company_info['company']}")
        st.markdown(f"**Industry:** {company_info['industry']}")
        st.markdown(f"**Description:** {company_info['description']}")

        st.markdown("**Offerings:**")
        if company_info["offerings"]:
            for item in company_info["offerings"]:
                st.markdown(f"- {item}")
        else:
            st.markdown("- (No detected offerings)")

        st.markdown("**Strategic Focus Areas:**")
        if company_info["strategic_focus"]:
            for area in company_info["strategic_focus"]:
                st.markdown(f"- {area}")
        else:
            st.markdown("- (No detected focus areas)")

        st.header("🚀 AI/GenAI Use Cases")
        # show use cases string (already bullet formatted)
        st.markdown(use_cases)

        # debug helper
        if st.checkbox("Show debug logs / last API errors"):
            st.write("Agent logs:")
            st.json({
                "research_agent_log": orchestrator.research_agent.log,
                "market_agent_log": orchestrator.market_agent.log
            })
            st.write("Last API errors (if any):")
            st.wri


