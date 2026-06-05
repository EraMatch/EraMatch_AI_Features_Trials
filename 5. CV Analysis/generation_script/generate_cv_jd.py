'''
https://www.kaggle.com/datasets/adhamashraf202200953/techncial-generated-job-descriptions?select=phase2_questions.jsonl
'''
import asyncio
import json
import os
from typing import List, Optional
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

class VerifiedSkill(BaseModel):
    skill: str
    confidence: str = Field(description="High, Medium, or Low")
    context: str = Field(description="Detailed proof from the CV showing how and where this skill was used.")

class GhostSkill(BaseModel):
    skill: str
    reason: str = Field(description="Why this is flagged as a ghost skill (e.g., listed but zero context in text).")

class SkillTriangulation(BaseModel):
    verified_skills: List[VerifiedSkill]
    ghost_skills: List[GhostSkill]

class EngineeringHygiene(BaseModel):
    score: str = Field(description="High, Medium, or Low")
    justification: str = Field(description="Detailed reason based on testing, code reviews, and standards mentioned in CV.")

class ProjectComplexity(BaseModel):
    highest_complexity_achievement: str = Field(description="The most technically complex thing the candidate built.")
    scale_and_impact_metrics: List[str] = Field(description="Any metrics, active sessions, latency reductions, or dataset sizes.")
    engineering_hygiene: EngineeringHygiene

class SeniorityAndGrowth(BaseModel):
    detected_seniority: str = Field(description="Junior, Mid-Level, Senior, or Lead")
    leadership_signals: List[str]
    career_trajectory: str = Field(description="Detailed paragraph analyzing the candidate's career growth and domain alignment.")

class ExpertAnalysisOutput(BaseModel):
    skill_triangulation: SkillTriangulation
    project_and_experience_complexity: ProjectComplexity
    seniority_and_growth: SeniorityAndGrowth

class SyntheticCVAndAnalysisPair(BaseModel):
    candidate_cv_text: str = Field(description="The full, richly detailed professional CV text with detailed work history and project descriptions.")
    expert_analysis: ExpertAnalysisOutput

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", "your-key"))

async def process_single_jd(jd_record: dict, candidate_type: str, semaphore: asyncio.Semaphore) -> Optional[dict]:
    async with semaphore:
        try:
            jd_text = jd_record.get("job_description", "")
            meta = jd_record.get("meta", {})
            evaluation_hash = jd_record.get("evaluation_hash", "unknown_hash")
            
            if not jd_text:
                return None

            if candidate_type == "PERFECT_FIT":
                type_instructions = """
                CANDIDATE TYPE: PERFECT FIT (Match Score: 85-95%)
                - The candidate must be highly qualified, explicitly meeting almost all required and preferred skills.
                - Their projects and work history must feature heavy technical depth, metrics, and deep alignment with the JD's domain.
                - Plant ONLY ONE minor Ghost Skill (a buzzword in skills but missing in text) to keep the expert analysis sharp.
                """
            elif candidate_type == "PARTIAL_FIT":
                type_instructions = """
                CANDIDATE TYPE: PARTIAL FIT / GAP PROFILE (Match Score: 55-70%)
                - The candidate is decent but has clear 'Skill Gaps'. They meet core requirements but lack preferred tools or domain context.
                - Plant 2 to 3 'Ghost Skills' (tools they claim to know in the skills section but have ZERO mention or usage in their actual project text or work history).
                - The Expert Analysis must explicitly catch these ghost skills and reflect a lower confidence score for them.
                """
            else: # SPAMMER / BUZZWORD STUFFER
                type_instructions = """
                CANDIDATE TYPE: BUZZWORD STUFFER / POOR FIT (Match Score: 20-40%)
                - The candidate is a CV spammer. They dumped advanced tools (e.g., Kubernetes, Kafka, Cloud Security) into their skills section to bypass ATS.
                - However, their actual work history and project text are highly generic, shallow, or only talk about basic features.
                - Plant 4+ 'Ghost Skills'. The Expert Analysis must severely flag this candidate, showing a massive mismatch between claimed skills and real context.
                """

            system_prompt = "You are an advanced technical recruiting AI specializing in synthetic data generation for resume parsing and deep screening."
            
            user_prompt = f"""
            You are generating training data. I will give you a Job Description and a specific Candidate Profile Type to simulate. 
            Your task is to generate a CV matching that profile type, AND the corresponding EXPERT ANALYSIS.

            TARGET JOB DESCRIPTION:
            {jd_text}

            TARGET METADATA CONTEXT:
            - Seniority Level: {meta.get('seniority', 'Not Specified')}
            - Tech Focus: {meta.get('tech_focus', 'Not Specified')}
            - Domain: {meta.get('domain', 'Not Specified')}

            --- PROFILE SPECIFIC INSTRUCTIONS ---
            {type_instructions}
            -------------------------------------

            GENERAL TEXT STANDARDS:
            - The CV MUST contain detailed professional sentences for Work History and Projects. Include library names, engineering bottlenecks, architectural choices, and exact metrics (e.g., active sessions, latency drops). Do NOT make it a short overview.
            """

            response = await client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=SyntheticCVAndAnalysisPair,
                temperature=0.7
            )
            
            generated_data = response.choices[0].message.parsed
            
            final_jsonl_line = {
                "evaluation_hash": evaluation_hash,
                "profile_strategy": candidate_type,اً
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a Technical Hiring AI. Analyze the provided Job Description and the Candidate's Parsed CV, then output a structured analysis evaluating skill validation, project complexity, and seniority."
                    },
                    {
                        "role": "user",
                        "content": f"JOB DESCRIPTION:\n{jd_text}\n\nCANDIDATE CV:\n{generated_data.candidate_cv_text}"
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(generated_data.expert_analysis.dict(), ensure_ascii=False)
                    }
                ]
            }
            return final_jsonl_line

        except Exception as e:
            print(f"Error processing record {jd_record.get('evaluation_hash', '')}: {str(e)}")
            return None

async def main():
    input_file = "phase1_job_descriptions.jsonl"
    output_file = "fine_tuning_ready_dataset.jsonl"
    
    MAX_CONCURRENT_TASKS = 5 
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    
    processed_hashes = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    line_data = json.loads(line)
                    if "evaluation_hash" in line_data:
                        processed_hashes.add(line_data["evaluation_hash"])
                except:
                    continue
        print(f"Found {len(processed_hashes)} already processed records. Skipping them.")

    all_valid_jds = []
    if not os.path.exists(input_file):
        print(f"Input file '{input_file}' not found!")
        return
        
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                if record.get("status") == "success":
                    all_valid_jds.append(record)
            except json.JSONDecodeError:
                continue

    total_available = len(all_valid_jds)
    print(f"Total valid JDs loaded from Phase 1: {total_available}")

    tasks = []
    skipped_count = 0
    
    for index, jd in enumerate(all_valid_jds):
        if jd.get("evaluation_hash") in processed_hashes:
            skipped_count += 1
            continue
            
        if index < 2000:
            candidate_type = "PERFECT_FIT"
        elif 2000 <= index < 4000:
            candidate_type = "PARTIAL_FIT"
        else:
            candidate_type = "SPAMMER"
            
        tasks.append(process_single_jd(jd, candidate_type, semaphore))

    total_to_process = len(tasks)
    print(f"Processing {total_to_process} pending items ({skipped_count} skipped via checkpoint).")

    count = 0
    with open(output_file, "a", encoding="utf-8") as out_f:
        for future in asyncio.as_completed(tasks):
            result = await future
            if result:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                count += 1
                if count % 10 == 0 or count == total_to_process:
                    print(f"Progress: {count}/{total_to_process} pairs generated and saved. Strategy used for last item: {result.get('profile_strategy')}")

    print(f"Training dataset successfully mapped and saved to '{output_file}'!")

if __name__ == "__main__":
    asyncio.run(main())
