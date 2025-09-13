import streamlit as st
import requests
from bs4 import BeautifulSoup
from typing import Dict, List
from groq import Groq

# -----------------------
# CONFIG
# -----------------------
MODEL_NAME = "llama-3.2-70b-versatile"   # fixed model

# Safe API key retrieval
api_key = st.secrets.get("GROQ_API_KEY")
if not api_key:
    st.error("❌ GROQ_API_KEY is missing. Please add it in .streamlit/secrets.toml")
    st.stop()

client = Groq(api_key=api_key)


# -----------------------
# WEB SCRAPER
# -----------------------
class WebBrowserTools:
    def __init__(self, model: str = MODEL_NAME):
        self.model = model

    def scrape_company_info(self, company_name: str) -> Dict:
        try:
            url = f"https://en.wikipedia.org/wiki/{company_name.replace(' ', '_')}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                para = soup.find("p")
                description = para.get_text().strip() if para else None
            else:
                description = None
        except Exception:
            description = None

        # If no description from wiki → use Groq
        if not description:
            description = self.generate_description_with_groq(company_name)

        offerings = self.generate_offerings(description)
        focus = self.generate_focus_areas(description)

        return {
            "company": company_name,
            "description": description,
            "offerings": offerings,
            "focus": focus,
        }

    def generate_description_with_groq(self, company_name: str) -> str:
        prompt = f"Write a 2-3 line professional description about {company_name}."
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

    def generate_offerings(self, description: str) -> List[str]:
        prompt = f"From this description, list the main offerings/products/services:\n\n{description}"
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return [item.strip("-• ") for item in response.choices[0].message.content.strip().split("\n") if item.strip()]

    def generate_focus_areas(self, description: str) -> List[str]:
        prompt = f"From this description, what strategic focus areas does the company have? (e.g., AI, Cloud, Healthcare)"
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return [item.strip("-• ") for item in response.choices[0].message.content.strip().split("\n") if item.strip()]


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
        description = company_info["description"]
        prompt = f"Suggest 5 AI/GenAI/ML use cases for the following company:\n\n{description}"
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
        )
        return [item.strip("-• ") for item in response.choices[0].message.content.strip().split("\n") if item.strip()]


class MultiAgentSystem:
    def __init__(self):
        self.research_agent = ResearchAgent()
        self.market_agent = MarketAnalysisAgent()

    def run(self, company_name: str) -> Dict:
        company_info = self.research_agent.research_company(company_name)
        use_cases = self.market_agent.generate_use_cases(company_info)
        return {"company_info": company_info, "use_cases": use_cases}


# -----------------------
# STREAMLIT APP
# -----------------------
st.set_page_config(page_title="AI Use Case Generator", page_icon="🤖")
st.title("🤖 Multi-Agent AI Use Case Generator")

company_name = st.text_input("Enter a company name to generate AI/ML/GenAI use cases and resources:")

if company_name:
    orchestrator = MultiAgentSystem()
    with st.spinner("Agents are working..."):
        results = orchestrator.run(company_name)

    st.success("✅ Completed!")
    info = results["company_info"]

    st.subheader("📄 Company Information")
    st.write(f"**Company:** {info['company']}")
    st.write(f"**Description:** {info['description']}")
    st.write("**Offerings:**")
    st.write(info["offerings"])
    st.write("**Strategic Focus Areas:**")
    st.write(info["focus"])

    st.subheader("🚀 AI/GenAI Use Cases")
    for case in results["use_cases"]:
        st.write("-", case)






