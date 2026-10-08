import os
import json
import random
import csv
import pandas as pd

project_root = r"c:\Users\OMKAR KUDAV\Desktop\my projects\resumescanner\Semantic-Resume-Screening-System"
job_fields_dir = os.path.join(project_root, "data", "job_fields")
resumes_dir = os.path.join(project_root, "data", "resumes")
csv_path = os.path.join(project_root, "data", "ranker_labels.csv")

NAMES = [
    "Rahul Sharma", "Aisha Gupta", "Michael Chen", "Sarah Jenkins", "Omkar Desai",
    "Priya Patel", "David Smith", "Elena Rodriguez", "Vikram Singh", "Tanya Ivanov",
    "Rohan Kumar", "Jessica Wong", "Amit Verma", "Sophia Martinez", "Karan Reddy",
    "John Doe", "Anita Roy", "James Wilson", "Neha Joshi", "William Taylor",
    "Arjun Nair", "Emily Brown", "Sanjay Mishra", "Olivia Davis", "Nikhil Rao",
    "Lucas Garcia", "Ananya Iyer", "Daniel Miller", "Kavya Menon", "Matthew Anderson",
    "Rajesh Gupta", "Meera Krishnan", "Liam O'Connor", "Chloe Dubois", "Aditya Sen",
    "Zoe Muller", "Wei Zhang", "Fatima Al-Fayed", "Carlos Santana", "Yuki Tanaka",
    "Siddharth Bose", "Maya Patel", "Ethan Hunt", "Mia Wallace", "Rishi Kapoor",
    "Isabella Rossi", "Ahmed Khan", "Sophie Martin", "Arun Jha", "Emma Watson",
    "Harsh Vardhan", "Lily Chen", "Omar Sharif", "Ava Williams", "Suresh Pillai",
    "Grace Lee", "Tariq Aziz", "Mia Johnson", "Vivek Sharma", "Ella Smith",
    "Ravi Teja", "Harper Brown", "Jamal Peters", "Abigail Jones", "Kishore Kumar",
    "Madison Davis", "Amir Khalid", "Elizabeth Taylor", "Tarun Goyal", "Avery Wilson",
    "Neha Sharma", "Sofia Garcia", "Pooja Hegde", "Mila Rodriguez", "Shruti Hassan",
    "Aria Martinez", "Kriti Sanon", "Camila Hernandez", "Disha Patani", "Luna Lopez",
    "Kiara Advani", "Aaliyah Gonzalez", "Anushka Sharma", "Layla Perez", "Deepika Padukone",
    "Riley Sanchez", "Priyanka Chopra", "Zoey Ramirez", "Katrina Kaif", "Nora Torres"
]

COLLEGES = [
    "Mumbai University", "Stanford University", "IIT Bombay", "MIT", "VJTI",
    "UC Berkeley", "Delhi University", "Georgia Tech", "NIT Trichy", "Cornell University",
    "Pune University", "University of Toronto", "BITS Pilani", "University of Washington",
    "SRM Institute of Science and Technology", "Imperial College London"
]

COMPANIES = [
    "TechNova", "GlobalSolutions", "Infosys", "DataMind Inc", "Apex Software",
    "TCS", "CloudScale", "Innovate LLC", "Wipro", "Alpha Analytics",
    "Cognizant", "CyberSecure Ltd", "NextGen Web", "Reliance Tech", "FinTech Systems",
    "Google", "Amazon", "Microsoft", "Meta", "Accenture", "Turing", "OpenAI"
]

PROJECTS = [
    "Built a scalable microservice architecture",
    "Designed and implemented a real-time data pipeline",
    "Developed a highly responsive user interface",
    "Led the migration to cloud infrastructure",
    "Created an automated testing framework",
    "Optimized database queries reducing latency by 40%",
    "Integrated third-party APIs for payment processing",
    "Trained machine learning models for predictive analytics",
    "Implemented security protocols and vulnerability patching",
    "Managed cross-functional agile development teams",
    "Architected distributed systems for high availability",
    "Automated deployment workflows using CI/CD pipelines"
]

TEMPLATE_PARAGRAPHS = [
    "I am a highly motivated professional with a track record of delivering quality results. My experience spans multiple projects where I have taken end-to-end responsibility. I thrive in collaborative environments and am always eager to learn new technologies.",
    "Experienced in leading teams and managing complex technical challenges. I have a strong foundation in core engineering principles and a passion for building scalable solutions. My background includes working closely with stakeholders to define requirements and deliver value.",
    "A results-driven individual with expertise in modern frameworks and tools. I have successfully contributed to several high-impact projects, demonstrating my ability to adapt and grow. I am focused on continuous improvement and writing clean, maintainable code.",
    "Driven by a desire to innovate and solve hard problems. I have hands-on experience across the entire software development lifecycle. I enjoy mentoring junior team members and fostering a culture of technical excellence.",
    "With a diverse background in various technical domains, I bring a unique perspective to problem-solving. I am committed to delivering robust and efficient systems. My goal is to leverage my skills to drive business success and technological advancement."
]

EXPERIENCE_LEVELS = [
    ("fresher", 0, 1),
    ("fresher", 0, 1),
    ("junior", 1, 3),
    ("mid", 3, 6),
    ("mid", 3, 6),
    ("senior", 6, 12),
    ("career-switcher", 1, 5),
    ("weak", 1, 4)
]

def get_synonym(skill):
    synonyms = {
        "machine learning": ["ML", "machine learning algorithms", "predictive modeling"],
        "artificial intelligence": ["AI", "machine intelligence"],
        "natural language processing": ["NLP", "text processing"],
        "large language models": ["LLMs", "foundation models"],
        "python": ["Python 3", "Python programming"],
        "javascript": ["JS", "ECMAScript"],
        "react": ["React.js", "ReactJS"],
        "node.js": ["Node", "NodeJS"],
        "aws": ["Amazon Web Services"],
        "gcp": ["Google Cloud Platform"],
        "sql": ["relational databases", "SQL Server", "MySQL"],
        "deep learning": ["neural networks", "DL"],
        "data analysis": ["data analytics", "data mining"],
        "ui/ux": ["user interface design", "user experience"],
        "cybersecurity": ["information security", "infosec"]
    }
    skill_lower = skill.lower()
    if skill_lower in synonyms:
        return random.choice([skill] + synonyms[skill_lower])
    return skill

def generate_resume_text(name, college, companies, profile_type, key_skills, description):
    skills_to_include = list(key_skills)
    match_quality = "Good"
    
    if profile_type == "career-switcher":
        num_skills = max(1, int(len(skills_to_include) * 0.55))
        skills_to_include = random.sample(skills_to_include, num_skills)
        match_quality = "Medium"
    elif profile_type == "weak":
        num_skills = max(1, len(skills_to_include) - 3)
        skills_to_include = random.sample(skills_to_include, num_skills)
        match_quality = "Poor"
        
    final_skills = [get_synonym(s) for s in skills_to_include]
    other_skills = ["Git", "Agile", "Communication", "Team Leadership", "Problem Solving", "Jira", "Docker", "Linux"]
    final_skills.extend(random.sample(other_skills, 3))
    random.shuffle(final_skills)
    
    email = name.lower().replace(" ", ".") + f"{random.randint(10,99)}@example.com"
    phone = f"+1-{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}"
    
    text = f"Name: {name}\n"
    text += f"Email: {email} | Phone: {phone}\n"
    text += f"Education: {college} - Graduated {random.randint(2018, 2024)}\n\n"
    
    text += "SUMMARY:\n"
    text += random.choice(TEMPLATE_PARAGRAPHS) + "\n\n"
    
    desc_words = description.split()
    pad_text = " ".join(random.sample(desc_words, min(len(desc_words), 50))) if desc_words else ""
    
    text += "SKILLS:\n"
    text += ", ".join(final_skills) + "\n\n"
    
    text += "EXPERIENCE:\n"
    num_jobs = random.randint(1, 3)
    current_year = 2026
    start_year = current_year - random.randint(1, 8)
    
    for i in range(num_jobs):
        end_year = start_year + random.randint(1, 3)
        if end_year > current_year:
            end_year = "Present"
        text += f"{random.choice(companies)} | {start_year} - {end_year}\n"
        text += f"- {random.choice(PROJECTS)} using {random.choice(final_skills) if final_skills else 'various tools'}.\n"
        text += f"- Collaborated with cross-functional teams to deliver on time.\n"
        if profile_type == "weak" and random.random() > 0.5:
            text += f"- Responsoble for mantaining legacy systems.\n"
        else:
            text += f"- {pad_text}\n"
        text += "\n"
        if end_year == "Present":
            break
        start_year = end_year
        
    word_count = len(text.split())
    while word_count < 350:
        extra_para = random.choice(TEMPLATE_PARAGRAPHS) + " " + " ".join(random.sample(desc_words, min(len(desc_words), 60))) if desc_words else random.choice(TEMPLATE_PARAGRAPHS)
        text += extra_para + "\n"
        word_count = len(text.split())
        
    return text, match_quality

def get_next_filename(slug, existing_files):
    for i in range(1, 100):
        fname = f"{slug}_{i:02d}.txt"
        if fname not in existing_files:
            return fname
    return f"{slug}_{random.randint(100,999)}.txt"

def main():
    if not os.path.exists(resumes_dir):
        os.makedirs(resumes_dir)
        
    existing_resumes = set(os.listdir(resumes_dir))
    job_fields = [f for f in os.listdir(job_fields_dir) if f.endswith('.json')]
    
    csv_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not csv_exists:
            writer.writerow(["filename", "true_field", "experience_level", "match_quality", "notes"])
            
        for jf in job_fields:
            slug = jf.replace('.json', '')
            with open(os.path.join(job_fields_dir, jf), 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            key_skills = data.get("key_skills", [])
            description = data.get("description", "")
            
            for exp_tuple in EXPERIENCE_LEVELS:
                profile_type = exp_tuple[0]
                
                name = random.choice(NAMES)
                NAMES.remove(name)
                if not NAMES:
                    NAMES.extend(["Unique Name " + str(x) for x in range(100)])
                    
                college = random.choice(COLLEGES)
                companies = random.sample(COMPANIES, 3)
                
                text, match_quality = generate_resume_text(name, college, companies, profile_type, key_skills, description)
                
                fname = get_next_filename(slug, existing_resumes)
                existing_resumes.add(fname)
                
                filepath = os.path.join(resumes_dir, fname)
                with open(filepath, 'w', encoding='utf-8') as text_file:
                    text_file.write(text)
                    
                writer.writerow([fname, slug, profile_type, match_quality, f"Synthetic {profile_type} resume"])
                
    final_resumes = [f for f in os.listdir(resumes_dir) if f.endswith('.txt')]
    print(f"\nTotal .txt files in data/resumes/: {len(final_resumes)}")
    
    df = pd.read_csv(csv_path)
    print(f"Row count of ranker_labels.csv: {len(df)}")
    print("\nCounts per true_field:")
    print(df['true_field'].value_counts())

if __name__ == "__main__":
    main()
