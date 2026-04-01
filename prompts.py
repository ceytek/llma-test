"""
Prompts adapted for Llama 3 8B.
Includes strict scoring rules, field-mismatch penalties, and few-shot examples.
"""

# ── CV Parsing ────────────────────────────────────────────────────────────────

CV_PARSE_SYSTEM = """You are a CV/Resume parser. You extract structured data from CVs written in any language (Turkish, English, German, etc.) and return ONLY valid JSON.

Rules:
- Return valid JSON only — no markdown, no explanation
- Use null for missing or unclear fields
- Dates must be in YYYY-MM format. "Haziran 2006" → "2006-06", "2001" → "2001-01"
- Preserve Turkish characters exactly: ı, ş, ğ, ü, ö, ç, İ, Ş, Ğ, Ü, Ö, Ç
- Language detection: if CV is written in Turkish set language to "TR", English → "EN", etc.
- For language skills: detect from CV content. If CV is in Turkish, the person is a native Turkish speaker.
- Extract ALL skills mentioned: technical, soft, tools, and spoken languages with levels."""

CV_PARSE_USER = """Parse this CV and return JSON with this exact structure:

{{
  "is_valid_cv": true,
  "language": "TR",
  "personal": {{
    "name": "full name or null",
    "email": "email or null",
    "phone": "phone or null",
    "location": "city/location or null"
  }},
  "summary": "brief professional summary or null",
  "education": [
    {{
      "degree": "Lisans/Yüksek Lisans/Doktora/Bachelor/Master/PhD",
      "field": "field of study",
      "institution": "school name",
      "graduation_year": 2020
    }}
  ],
  "experience": [
    {{
      "title": "job title",
      "company": "company name",
      "start_date": "YYYY-MM",
      "end_date": "YYYY-MM or present",
      "description": "key responsibilities and achievements"
    }}
  ],
  "skills": {{
    "technical": ["programming languages, frameworks, cloud platforms, data tools"],
    "soft": ["leadership, communication, project management"],
    "languages": [
      {{"language": "Turkish", "level": "Native"}},
      {{"language": "English", "level": "Advanced/Intermediate/Basic"}}
    ],
    "tools": ["software, platforms, IDEs"]
  }},
  "total_experience_years": 5.0
}}

IMPORTANT for language skills:
- If the CV is written in Turkish → add {{"language": "Turkish", "level": "Native"}}
- Look for explicit language mentions like "İngilizce - İleri Seviye" or "English - Advanced"
- Map levels: Ana dil/Anadil = Native, İleri = Advanced, Orta = Intermediate, Başlangıç = Basic
- If no English skill is mentioned anywhere in the CV, do NOT add English to languages list

CV TEXT:
{cv_text}

Return ONLY the JSON object."""


# ── Job–CV Matching ───────────────────────────────────────────────────────────

MATCH_SYSTEM = """You are a strict, objective HR analyst. You score candidates ONLY based on concrete evidence from their CV. You must be harsh when the candidate's background does not match the job field. Do NOT be generous. Return ONLY valid JSON.

CRITICAL SCORING RULES:
- If the candidate's career field is COMPLETELY DIFFERENT from the job field, overall_score MUST be below 30.
- If the candidate lacks most required technical skills, skills_score MUST be below 10.
- Years of experience in an UNRELATED field count for almost nothing (max 5 out of 30).
- A degree in an unrelated field scores low (max 5 out of 20).
- Do NOT give points for "transferable skills" unless they are explicitly required in the job posting.
- Be realistic: an HR specialist applying for a Data Engineer role should score below 25 total.

TURKISH LANGUAGE RULES for all text output:
- Write in natural, fluent Turkish. Use simple sentence structures.
- CORRECT: "Adayın deneyimi pozisyonla uyuşmuyor."
- WRONG: "Adayın deneyimi, pozisyonun gereksinmelerine karşılık gelmemektedir."
- Use everyday Turkish, avoid overly formal or machine-translated phrasing."""

MATCH_USER = """Analyze this candidate against the job requirements. Be STRICT and OBJECTIVE.

**JOB POSTING:**
Title: {job_title}
Department: {job_department}
Location: {job_location}
Experience Required: {job_experience}
Education Required: {job_education}
Description: {job_description}
Required Skills/Keywords: {job_skills}

**CANDIDATE CV DATA:**
Name: {candidate_name}
Location: {candidate_location}
Total Experience: {candidate_experience} years
Education: {candidate_education}
Technical Skills: {candidate_skills}
Work History:
{candidate_experience_detail}

---

STRICT SCORING GUIDE (read carefully before scoring):

experience_score (0-30):
  - 25-30: Candidate has 3+ years doing the EXACT same role with matching technologies
  - 15-24: Candidate has relevant experience in a SIMILAR technical field
  - 5-14: Candidate has some overlapping experience but mostly different field
  - 0-4: Candidate's experience is in a COMPLETELY DIFFERENT field (e.g., HR vs Engineering)

education_score (0-20):
  - 15-20: Degree directly matches (e.g., Computer Science for a Software Engineer role)
  - 8-14: Degree is somewhat related (e.g., Mathematics for a Data role)
  - 0-7: Degree is unrelated (e.g., Literature for an Engineering role)

skills_score (0-30):
  - 25-30: Candidate has 80%+ of required skills
  - 15-24: Candidate has 50-80% of required skills
  - 5-14: Candidate has 20-50% of required skills
  - 0-4: Candidate has less than 20% of required skills

language_score (0-10):
  - 8-10: All required language skills met or exceeded
  - 4-7: Some language requirements met
  - 0-3: Key language requirements not met

fit_score (0-10):
  - 8-10: Career trajectory clearly leads to this role
  - 4-7: Some alignment in career direction
  - 0-3: Career path is in a completely different direction

recommendation thresholds:
  - highly_recommended: overall_score >= 75
  - recommended: overall_score 50-74
  - maybe: overall_score 30-49
  - not_recommended: overall_score < 30

--- FEW-SHOT EXAMPLES ---

EXAMPLE 1: GOOD MATCH (Software Developer job ↔ Software Developer CV)
Job: "Senior Python Developer", Required: Python, Django, PostgreSQL, Docker
Candidate: 5 years Python/Django experience, CS degree, knows Docker and PostgreSQL
Result: {{"overall_score": 82, "recommendation": "highly_recommended"}}

EXAMPLE 2: PARTIAL MATCH (Data Engineer job ↔ Backend Developer CV)
Job: "Data Engineer", Required: Python, BigQuery, Airflow, Spark
Candidate: 4 years backend Python, knows SQL, no BigQuery/Spark/Airflow experience
Result: {{"overall_score": 42, "recommendation": "maybe"}}

EXAMPLE 3: BAD MATCH (Data & AI Engineer job ↔ HR Specialist CV)
Job: "Data & AI Engineer", Required: Python, BigQuery, ML, Cloud platforms
Candidate: 20 years HR experience, PhD in Labor Economics, knows MS Office and basic SQL
Result: {{"overall_score": 12, "recommendation": "not_recommended"}}

--- END EXAMPLES ---

Now score this candidate. Return JSON:
{{
  "overall_score": <integer 0-100>,
  "recommendation": "<highly_recommended|recommended|maybe|not_recommended>",
  "breakdown": {{
    "experience_score": <integer 0-30>,
    "experience_reasoning": "<1-2 sentences in Turkish>",
    "education_score": <integer 0-20>,
    "education_reasoning": "<1-2 sentences in Turkish>",
    "skills_score": <integer 0-30>,
    "skills_reasoning": "<1-2 sentences in Turkish>",
    "language_score": <integer 0-10>,
    "language_reasoning": "<1-2 sentences in Turkish>",
    "fit_score": <integer 0-10>,
    "fit_reasoning": "<1-2 sentences in Turkish>"
  }},
  "matched_skills": ["only skills that ACTUALLY match"],
  "missing_skills": ["required skills NOT found in CV"],
  "strengths": ["max 3, in Turkish"],
  "weaknesses": ["max 3, in Turkish"],
  "summary": "<2-3 sentences in natural Turkish summarizing the match>"
}}

overall_score MUST equal the sum of all breakdown scores.
Return ONLY the JSON object."""


def build_cv_parse_prompt(cv_text: str) -> str:
    return CV_PARSE_USER.replace("{cv_text}", cv_text)


def build_match_prompt(job: dict, candidate: dict) -> str:
    edu_summary = ""
    if candidate.get("education"):
        for e in candidate["education"]:
            if isinstance(e, dict):
                edu_summary += f"{e.get('degree', '')} - {e.get('field', '')} @ {e.get('institution', '')}\n"

    exp_summary = ""
    if candidate.get("experience"):
        for e in candidate["experience"]:
            if isinstance(e, dict):
                desc = e.get("description", "")
                desc_short = desc[:150] + "..." if desc and len(desc) > 150 else desc or ""
                exp_summary += (
                    f"- {e.get('title', '')} @ {e.get('company', '')} "
                    f"({e.get('start_date', '?')} – {e.get('end_date', '?')}): "
                    f"{desc_short}\n"
                )

    skills_list = []
    sk = candidate.get("skills", {})
    if isinstance(sk, dict):
        for category in ("technical", "soft", "tools"):
            items = sk.get(category, [])
            if isinstance(items, list):
                skills_list.extend(str(s) for s in items)

    lang_summary = ""
    if isinstance(sk, dict) and sk.get("languages"):
        for lang in sk["languages"]:
            if isinstance(lang, dict):
                lang_summary += f"{lang.get('language', '?')}: {lang.get('level', '?')}, "
    lang_summary = lang_summary.rstrip(", ") or "Belirtilmemiş"

    candidate_skills_str = ", ".join(skills_list) if skills_list else "Belirtilmemiş"

    return MATCH_USER.format(
        job_title=job.get("title", "N/A"),
        job_department=job.get("department", "N/A"),
        job_location=job.get("location", "N/A"),
        job_experience=job.get("experience_level", "N/A"),
        job_education=job.get("required_education", "N/A"),
        job_description=job.get("description", "N/A"),
        job_skills=", ".join(job.get("keywords", [])) if job.get("keywords") else "N/A",
        candidate_name=candidate.get("personal", {}).get("name", "N/A"),
        candidate_location=candidate.get("personal", {}).get("location", "N/A"),
        candidate_experience=candidate.get("total_experience_years", "N/A"),
        candidate_education=edu_summary.strip() or "Belirtilmemiş",
        candidate_skills=f"{candidate_skills_str}\nSpoken Languages: {lang_summary}",
        candidate_experience_detail=exp_summary.strip() or "Belirtilmemiş",
    )
