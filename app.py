import streamlit as st
import requests
from typing import Dict, List
from groq import Groq

# Configure Groq client
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

class BaseAgent:
    def __init__(self, name: str):
        self.name = name
        self.log = []

    def log_action(self, action: str):
        self.log.append(f"{self.name}: {action}")

class WebBrowserTools:
    def __init__(self):
        # Using Groq model (you can change to another supported model, e.g., "llama-3.1-70b-versatile")
        self.model = "llama-3.1-70b-versatile"

    def scrape_company_info(self, company_name: str) -> Dict:
        try:
            company_name_formatted = company_name.replace(' ', '_')
            response = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{company_name_formatted}")
            if response.status_code == 200:
                data = response.json()
                description = data.get("extract", "")
                if not description or "may refer to:" in description:
                    description = self.generate_description_with_groq(company_name)
            else:
                description = self.generate_description_with_groq(company_name)
        except Exception:
            description = self.generate_description_with_groq(company_name)

        offerings = self.generate_offerings(description)
        strategic_focus = self.generate_focus_areas(description)

        return {
            "description": description,
            "products": offerings,
            "focus_areas": strategic_focus
        }

    def generate_description_with_groq(self, company_name: str) -> str:
        prompt = f"Please write a 2-3 line professional description about the company '{company_name}'."
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()

    def generate_focus_areas(self, description: str) -> List[str]:
        prompt = f"Based on the following description, suggest 2-3 strategic focus areas for the company:\n\n{description}"
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        focus_text = response.choices[0].message.content.strip()
        return [area.strip("- ") for area in focus_text.split("\n") if area.strip()]

    def generate_offerings(self, description: str) -> List[str]:
        prompt = f"Based on the following company description, list 2-3 main products or services they offer:\n\n{description}"
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        offerings_text = response.choices[0].message.content.strip()
        return [item.strip("- ") for item in offerings_text.split("\n") if item.strip()]

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
        description = description.lower()
        if "automotive" in description or "vehicle" in description:
            return "Automotive"
        elif "finance" in description or "bank" in description:
            return "Finance"
        elif "e-commerce" in description or "retail" in description:
            return "E-commerce/Retail"
        elif "technology" in description or "software" in description:
            return "Technology"
        elif "entertainment" in description or "media" in description:
            return "Entertainment/Media"
        elif "healthcare" in description or "medical" in description:
            return "Healthcare"
        else:
            return "General Industry"

class MarketAnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__("Market Analysis Agent")
        self.model = "llama-3.1-70b-versatile"

    def generate_use_cases(self, industry_data: Dict) -> str:
        self.log_action("Generating AI/ML/GenAI use cases")
        prompt = f"""You are an AI business consultant.
Analyze the following industry and suggest AI/ML and Generative AI (GenAI) use cases:

Industry: {industry_data['industry']}
Description: {industry_data['description']}
Offerings: {industry_data['offerings']}
Strategic Focus Areas: {industry_data['strategic_focus']}

Please suggest:
- Practical AI/ML/GenAI solutions
- Improvements to customer experience, operations, supply chain
- Internal GenAI solutions (chatbots, automated reporting, document search)

Respond with a readable bullet-point format.
"""
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()

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

# Streamlit UI
st.set_page_config(page_title="Multi-Agent AI Use Case Generator", page_icon="🤖")
st.title("🤖 Multi-Agent AI Use Case Generator")
st.write("Enter a company name to generate AI/ML/GenAI use cases and resources:")

company_name = st.text_input("Company Name")

if st.button("Run Agents 🚀"):
    if company_name:
        orchestrator = MultiAgentSystem()
        with st.spinner('Agents are working...'):
            results = orchestrator.run(company_name)
        st.success('✅ Completed!')

        company_info = results['company_info']
        use_cases = results['ai_use_cases']

        st.header("📄 Company Information")
        st.markdown(f"**Company:** {company_info['company']}")
        st.markdown(f"**Industry:** {company_info['industry']}")
        st.markdown(f"**Description:** {company_info['description']}")
        st.markdown(f"**Offerings:**")
        for item in company_info['offerings']:
            st.markdown(f"- {item}")
        st.markdown(f"**Strategic Focus Areas:**")
        for area in company_info['strategic_focus']:
            st.markdown(f"- {area}")

        st.header("🚀 AI/GenAI Use Cases")
        st.markdown(use_cases)

    else:
        st.warning("Please enter a company name first!")
