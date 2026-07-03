# Medical RAG: Embedding & QA Dataset Pipeline

This repository contains the pipeline for generating a specialized medical Question-Answering (QA) dataset using LLM-assisted extraction, alongside the integration details for the fine-tuned medical embedding model.

For a deep dive into the underlying architecture, training data, and specific evaluations, check out the official model card on Hugging Face:
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Card-yellow)](https://huggingface.co/Gaykar/all-MiniLM-L6-medical-rag)

---

## 🛠️ Tech Stack

* **Embedding & NLP:** `sentence-transformers`
* **Data Manipulation:** `pandas`, `numpy`
* **Visualization & Analytics:** `matplotlib`
* **LLM Orchestration:** `langchain`, `Groq API`

---

## 🧠 Training Details & Dataset Creation Pipeline

📊 **Dataset Purpose:** This dataset was created purely for academic and learning purposes to demonstrate skills in data collection, preprocessing, and LLM-based data generation within the medical NLP domain. 

> ⚠️ **Compliance Note:** No proprietary or copyrighted text is redistributed in raw form. All source material was used only as temporary input to generate transformed outputs.

The overall pipeline consists of two stages:

### 1️⃣ Stage 1: Data Collection (Source Material)
Medical information related to brain tumors and human health was gathered from openly accessible educational and public medical resources. These sources were used strictly as intermediate context to generate derived question–answer pairs.

#### Sources Used
* **Public Medical Websites:** *Example: Mayo Clinic* (Used only to understand structure and terminology; no verbatim content stored).
* **Educational Textbooks (Open-Access PDFs):**
  * NCERT Biology (Human Health and Disease) — [Source Link](https://ncert.nic.in/textbook/pdf/lebo107.pdf)
  * NIOS Senior Secondary Biology — [Source Link](https://nios.ac.in/media/documents/SrSec314NewE/Lesson-29.pdf)
* **Open-Source QA Dataset:**
  * MedQuAD (Medical Question Answer Dataset) — [Hugging Face Link](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset) (~6,000 samples already structured as QA pairs).

---

### 2️⃣ Stage 2: Data Formatting & Generation
To convert unstructured medical text into structured data, an LLM-assisted pipeline was implemented using **LangChain** and the **Groq API**.

#### Workflow
1. **Extraction:** Pulled raw medical text chunks from websites and PDFs.
2. **Contextual Feeding:** Passed extracted text chunks directly as context to the LLM.
3. **Synthesis:** Prompted the LLM to generate high-quality question–answer pairs.
4. **Filtering (Discarded):** * Non-medical content.
   * Questions without valid answers.
   * Low-information or irrelevant text.
5. **Storage:** Stored *only* the successfully validated, generated QA pairs in JSON format.

#### Prompt Design
```python
prompt = PromptTemplate(
    input_variables=["page_content"],
    template="""You are a medical AI assistant.
Given the following medical text, generate high-quality question and answer pairs.
Ignore any non-medical information.

IMP: Ignore if the data only contains questions without answers. Do not generate questions from such data.

Rules:
- Use ONLY the provided content
- Ignore sentences that do not contain meaningful medical information
- Do NOT hallucinate
- If no useful information exists, return an empty list
- Output MUST be valid JSON only

Output format:
[
  {
    "question": "...",
    "answer": "..."
  }
]

Medical Text:
{page_content}"""
)
