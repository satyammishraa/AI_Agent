# app.py (updated parts only)
import streamlit as st
import requests
from bs4 import BeautifulSoup   # new import

class WebBrowserTools:
    def __init__(self, model: str = MODEL_NAME):
        self.model = model

    def scrape_company_info(self, company_name: str) -> Dict:
        """Try multiple fallbacks to get a reliable description."""
        description = self._fetch_wikipedia_summary(company_name)

        if not description:  # fallback to LLM if wiki fails
            description = self.generate_description_with_groq(company_name)

        if not description:  # ultimate fallback
            description = f"{company_name} is a well-known company operating globally."

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

    def _fetch_wikipedia_summary(self, company_name: str) -> str:
        """First try the official Wikipedia API, then fallback to scraping HTML."""
        try:
            company_name_formatted = company_name.replace(" ", "_")
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{company_name_formatted}"
            resp = requests.get(url, timeout=8)

            if resp.status_code == 200:
                data = resp.json()
                desc = data.get("extract", "")
                if desc and "may refer to" not in desc.lower():
                    return desc
        except Exception as e:
            print("Wiki summary API error:", e)

        # Fallback: scrape first paragraph from Wikipedia HTML
        try:
            html_url = f"https://en.wikipedia.org/wiki/{company_name.replace(' ', '_')}"
            html_resp = requests.get(html_url, timeout=8)
            if html_resp.status_code == 200:
                soup = BeautifulSoup(html_resp.text, "html.parser")
                p_tags = soup.select("p")
                for p in p_tags:
                    text = p.get_text(strip=True)
                    if text and len(text) > 50 and "may refer to" not in text.lower():
                        return text
        except Exception as e:
            print("Wiki HTML scrape error:", e)

        return ""

    def generate_description_with_groq(self, company_name: str) -> str:
        """Fallback LLM description generator"""
        prompt = f"Write a concise 2-3 line professional description about the company '{company_name}'."
        try:
            text = self._call_chat_completion(
                prompt,
                system_role="You are a concise business copywriter.",
                max_tokens=180
            )
            return text.strip()
        except Exception as e:
            st.session_state.setdefault("last_api_error", [])
            st.session_state["last_api_error"].append(str(e))
            print("Groq API Error (description):", e)
            return ""   # return empty → triggers wiki fallback



