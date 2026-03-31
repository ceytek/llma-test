"""
Prompts adapted for Llama 3 8B.
Kept shorter and more direct than the OpenAI variants to fit smaller context.
"""

# ── CV Parsing ────────────────────────────────────────────────────────────────

CV_PARSE_SYSTEM = """You are a CV/Resume parser. Extract structured data from CVs in any language (Turkish, English, etc.) and return ONLY valid JSON.

Rules:
- Return valid JSON, nothing else
- Use null for missing fields
- Dates in YYYY-MM format
- Preserve Turkish characters (ı, ş, ğ, ü, ö, ç, İ)"""

CV_PARSE_USER = """Parse this CV and return JSON with this exact structure:

{{
  "is_valid_cv": true,
  "personal": {{
    "name": "full name or null",
    "email": "email or null",
    "phone": "phone or null",
    "location": "city/location or null"
  }},
  "summary": "brief professional summary or null",
  "education": [
    {{
      "degree": "degree level",
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
      "description": "responsibilities"
    }}
  ],
  "skills": {{
    "technical": ["skill1", "skill2"],
    "soft": ["skill1"],
    "languages": [
      {{"language": "Turkish", "level": "Native"}}
    ],
    "tools": ["tool1"]
  }},
  "total_experience_years": 5.0
}}

CV TEXT:
{cv_text}

Return ONLY the JSON object."""


# ── Job–CV Matching ───────────────────────────────────────────────────────────

MATCH_SYSTEM = """You are an expert HR analyst. Compare a candidate's CV against job requirements and score objectively. Return ONLY valid JSON."""

MATCH_USER = """Analyze this candidate against the job and return a matching score.

**JOB:**
Title: {job_title}
Department: {job_department}
Location: {job_location}
Experience Required: {job_experience}
Education Required: {job_education}
Description: {job_description}
Required Skills: {job_skills}

**CANDIDATE:**
Name: {candidate_name}
Location: {candidate_location}
Total Experience: {candidate_experience} years
Education: {candidate_education}
Skills: {candidate_skills}
Experience: {candidate_experience_detail}

Return JSON with this exact structure:
{{
  "overall_score": 75,
  "recommendation": "recommended",
  "breakdown": {{
    "experience_score": 22,
    "experience_reasoning": "brief explanation",
    "education_score": 15,
    "education_reasoning": "brief explanation",
    "skills_score": 23,
    "skills_reasoning": "brief explanation",
    "language_score": 7,
    "language_reasoning": "brief explanation",
    "fit_score": 8,
    "fit_reasoning": "brief explanation"
  }},
  "matched_skills": ["skill1", "skill2"],
  "missing_skills": ["skill3"],
  "strengths": ["strength1", "strength2"],
  "weaknesses": ["weakness1"],
  "summary": "2-3 sentence summary in Turkish"
}}

Scoring weights: experience 0-30, education 0-20, skills 0-30, language 0-10, fit 0-10 = max 100.
recommendation must be one of: highly_recommended, recommended, maybe, not_recommended
Write summary, strengths, weaknesses, and reasoning fields in Turkish.
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
                exp_summary += (
                    f"- {e.get('title', '')} @ {e.get('company', '')} "
                    f"({e.get('start_date', '?')} – {e.get('end_date', '?')})\n"
                )

    skills_list = []
    sk = candidate.get("skills", {})
    if isinstance(sk, dict):
        for category in ("technical", "soft", "tools"):
            items = sk.get(category, [])
            if isinstance(items, list):
                skills_list.extend(str(s) for s in items)

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
        candidate_education=edu_summary.strip() or "N/A",
        candidate_skills=", ".join(skills_list) or "N/A",
        candidate_experience_detail=exp_summary.strip() or "N/A",
    )
