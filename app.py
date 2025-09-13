import streamlit as st
import requests
from groq import Groq
from bs4 import BeautifulSoup

# ----------------------------
# CONFIG
# ----------------------------
MODEL_NAME = "llama-3.1-70b-versatile"

# Initialize Groq client
client = Groq(api_key=st.secrets["GROQ_API_KEY"])


# ----------------------------
# TOOLS
# ----------------------------
class WebBrowserTools:
    def __init__(self, model: str = MODEL_NAME):
        self.model = model

    def scrape_company_info(self, company_name: str) -> str:
        """Try Wikipedia API, fallback to scraping"""
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{company_name}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "extract" in data:
                    return data["extract"]
        except Exception:
            pass

        # fallback to scraping HTML
        try:
            html_url = f"https://en.wikipedia.org/wiki/{company_name}"
            res = requests.get(html_url, timeout=10)
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, "html.parser")
                paragraphs = soup.select("p")
                if paragraphs:
                    return paragraphs[0].get_text().strip()
        except Exception:
            pass

        return ""  # nothing found


class ResearchAgent:
    def __init__(self, browser_tools: WebBrowserTools, model: str = MODEL_NAME):
        self.browser_tools = browser_tools
        self.model = model

    def research_company(self, company_name: str) -> dict:
        description = self.browser_tools.scrape_company_info(company_name)
        if not description:
            description = "No reliable description found."

        return {
            "company": company_name,
            "industry": "General Industry",
            "description": description,
            "offerings": [description] if description else [],
            "strategic_focus": ["AI / Machine Learning"],
        }


class UseCaseAgent:
    def __init__(self, model: str = MODEL_NAME):
        self.model = model

    def generate_use_cases(self, company_info: dict) -> list:
        prompt = f"""
        You are an AI strategist. Based on the following company profile, 
        generate 5 concrete AI/ML/GenAI use cases that align with its business.

        Company Info:
        Name: {company_info['company']}
        Industry: {company_info['industry']}
        Description: {company_info['description']}
        Offerings: {', '.join(company_info['offerings'])}
        Strategic Focus Areas: {', '.join(company_info['strategic_focus'])}
        """

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500,
            )
            content = response.choices[0].message.content
            return content.split("\n")
        except Exception:
            return ["❌ API error — could not generate use cases."]


# ----------------------------
# ORCHESTRATOR
# ----------------------------
class Orchestrator:
    def __init__(self):
        self.browser_tools = WebBrowserTools()
        self.research_agent = ResearchAgent(self.browser_tools)
        self.use_case_agent = UseCaseAgent()

    def run(self, company_name: str):
        company_info = self.research_agent.research_company(company_name)
        use_cases = self.use_case_agent.generate_use_cases(company_info)
        return company_info, use_cases


# ----------------------------
# STREAMLIT APP
# ----------------------------
st.set_page_config(page_title="🤖 Multi-Agent AI Use Case Generator", layout="wide")

st.title("🤖 Multi-Agent AI Use Case Generator")
st.write("Enter a company name to generate AI/ML/GenAI use cases and resources:")

company_name = st.text_input("Company Name")

if st.button("Generate Use Cases") and company_name:
    orchestrator = Orchestrator()
    with st.spinner("🔎 Researching company and generating ideas..."):
        company_info, use_cases = orchestrator.run(company_name)

    st.success("✅ Completed!")

    st.subheader("📄 Company Information")
    st.write(f"**Company:** {company_info['company']}")
    st.write(f"**Industry:** {company_info['industry']}")
    st.write(f"**Description:** {company_info['description']}")

    if company_info["offerings"]:
        st.write("**Offerings:**")
        for o in company_info["offerings"]:
            st.write(f"- {o}")

    st.write("**Strategic Focus Areas:**")
    for f in company_info["strategic_focus"]:
        st.write(f"- {f}")

    st.subheader("🚀 AI/GenAI Use Cases")
    for uc in use_cases:
        if uc.strip():
            st.write(f"- {uc.strip()}")




