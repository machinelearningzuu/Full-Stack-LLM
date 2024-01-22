import os, arxiv
import nltk, json, PyPDF2
from rake_nltk import Rake

nltk.download("stopwords")
nltk.download("punkt")

def refine_query(query):
    rake = Rake()
    rake.extract_keywords_from_text(query)
    keywords = rake.get_ranked_phrases()
    return " ".join(keywords)

def scrape_papers(
                query,
                max_results=10
                ):
    
    if len(os.listdir("data/pdf")) > 0:
        for f in os.listdir("data/pdf"):
            os.remove(f"data/pdf/{f}")

    if len(os.listdir("data/json")) > 0:
        for f in os.listdir("data/json"):
            os.remove(f"data/json/{f}")
            
    refined_query = refine_query(query)
    results = []

    search = arxiv.Search(
                        query=refined_query,
                        max_results=max_results,
                        sort_by=arxiv.SortCriterion.Relevance,
                        )
    papers = list(search.results())

    for i, p in enumerate(papers):
        text = ""
        file_path = f"data/pdf/data_{i}.pdf"
        p.download_pdf(filename=file_path)

        with open(f"data/pdf/data_{i}.pdf", "rb") as file:
            pdf = PyPDF2.PdfReader(file)

            for page in range(len(pdf.pages)):
                page_obj = pdf.pages[page]

                text += page_obj.extract_text() + " "

        paper_doc = {"url": p.pdf_url, "title": p.title, "text": text}
        results.append(paper_doc)

    for i, r in enumerate(results):
        with open(f"data/json/data_{i}.json", "w") as f:
            json.dump(r, f)

    return results