import arxiv
from datetime import datetime, timedelta
import pytz
import requests
import numpy as np


def print_papers(papers):
    if not papers:
        print(
            "No papers found with the given keywords in the title for the specified date range."
        )
        return

    for paper in papers:
        print(f"Title: {paper['title']}")
        print(f"Authors: {paper['authors']}")
        print(f"Published: {paper['published']}")
        print(f"URL: {paper['url']}")
        print()
    return


def search_arxiv_papers(keywords, start_date, end_date, max_results=100):
    # Convert start_date and end_date to timezone-aware UTC datetime objects
    start_date = pytz.utc.localize(start_date)
    end_date = pytz.utc.localize(end_date)

    # Perform the search on arXiv within the date range
    client = arxiv.Client()
    search_query = " OR ".join([f'ti:"{keyword}"' for keyword in keywords])
    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    # Filter results where the title contains the keywords and is within the date range
    papers = []
    for result in client.results(search):
        if start_date <= result.published <= end_date:
            title_lower = result.title.lower()
            if any(keyword.lower() in title_lower for keyword in keywords):
                paper_info = {
                    "title": result.title,
                    "authors": ", ".join(author.name for author in result.authors),
                    "published": result.published.strftime("%Y-%m-%d"),
                    "url": result.entry_id,
                }
                papers.append(paper_info)

    return papers


def generate_github_gist(papers, github_token):
    # Combine paper details into a single Markdown-formatted text
    gist_content = "\n\n".join(
        [
            f"### [{paper['title']}]({paper['url']})\n**Authors:** {paper['authors']}\n**Published:** {paper['published']}"
            for paper in papers
        ]
    )

    # Create a GitHub Gist with the content
    gist_data = {
        "description": "Latest Research Papers",
        "public": True,
        "files": {"latest_research_papers.md": {"content": gist_content}},
    }

    headers = {"Authorization": f"token {github_token}"}

    response = requests.post(
        "https://api.github.com/gists", json=gist_data, headers=headers
    )

    if response.status_code == 201:
        gist_url = response.json()["html_url"]
        return gist_url
    else:
        return f"Error creating GitHub Gist: {response.status_code} {response.text}"


def main():
    # Define the keywords to search in the title
    keywords = [
        "Natural language processing",
        "transformer",
        "large language model",
        "causal inference",
        "Neural networks",
        "Causality",
    ]

    # Define the date range for the search (last 7 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    # Define the maximum number of results to retrieve
    max_results = 10

    # Search for papers within the date range
    papers = search_arxiv_papers(
        keywords, start_date, end_date, max_results=max_results
    )

    print_papers(papers)
    # If papers are found, generate a GitHub Gist link with the results
    if papers:
        github_token = ""  # Replace with your personal access token
        gist_link = generate_github_gist(papers, github_token)
        print(f"Here is your single shareable link for the latest papers:\n{gist_link}")
    else:
        print(
            "No papers found with the given keywords in the title for the specified date range."
        )


if __name__ == "__main__":
    main()
