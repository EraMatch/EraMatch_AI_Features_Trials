"""cv_generator_batch2.py — Batch 2 Synthetic CV Generator.

Generates CVSchema v3.0 objects using Faker + template-based content.
No LLM calls, no GitHub API, no real emails.

Data pools and generation logic extracted from cvs-eramatch.ipynb.
"""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dateutil.relativedelta import relativedelta
from faker import Faker

from schema import (
    CertificationEntry,
    ContactInfo,
    CVSchema,
    EducationEntry,
    MiscItem,
    ProjectEntry,
    SkillEntry,
    WorkExperienceEntry,
)

# ---------------------------------------------------------------------------
# Data Pools (from cvs-eramatch.ipynb)
# ---------------------------------------------------------------------------

DOMAINS: Dict[str, Dict[str, Any]] = {
    "backend": {
        "titles": [
            "Backend Developer",
            "Software Engineer",
            "API Developer",
            "Platform Engineer",
            "Server-Side Developer",
            "AI Engineer",
            "LLM Engineer",
        ],
        "skills": {
            "programming_language": [
                "Python",
                "Java",
                "Go",
                "Rust",
                "Node.js",
                "C#",
                "Ruby",
                "PHP",
                "Kotlin",
                "TypeScript",
            ],
            "framework": [
                "Django",
                "FastAPI",
                "Spring Boot",
                "Express.js",
                "Flask",
                "NestJS",
                "Rails",
                "ASP.NET",
                "Gin",
                "Actix",
                "LangChain",
                "LlamaIndex",
            ],
            "database": [
                "PostgreSQL",
                "MySQL",
                "MongoDB",
                "Redis",
                "DynamoDB",
                "Cassandra",
                "SQLite",
                "Elasticsearch",
                "CockroachDB",
                "Neon",
                "PlanetScale",
                "Supabase",
            ],
            "tool": [
                "Docker",
                "Git",
                "Kubernetes",
                "Jenkins",
                "Terraform",
                "Ansible",
                "Nginx",
                "RabbitMQ",
                "Kafka",
                "gRPC",
                "Helm",
                "ArgoCD",
                "GitHub Actions",
            ],
            "cloud": [
                "AWS",
                "GCP",
                "Azure",
                "Heroku",
                "DigitalOcean",
                "Vercel",
                "Fly.io",
                "Railway",
                "Render",
                "Cloudflare",
            ],
            "ai_api": [
                "OpenAI API",
                "Claude API",
                "Anthropic",
                "Google Gemini",
                "AWS Bedrock",
                "Azure OpenAI",
                "Hugging Face",
                "Replicate",
            ],
            "methodology": [
                "Agile",
                "Scrum",
                "CI/CD",
                "TDD",
                "Microservices",
                "REST API Design",
                "GraphQL",
                "Event-Driven Architecture",
                "LLM Integration",
                "RAG Architecture",
            ],
        },
        "primary_domain": "Backend Engineering",
    },
    "frontend": {
        "titles": [
            "Frontend Developer",
            "UI Engineer",
            "React Developer",
            "Full-Stack Developer",
            "Web Developer",
        ],
        "skills": {
            "programming_language": [
                "JavaScript",
                "TypeScript",
                "HTML",
                "CSS",
                "Python",
                "Dart",
            ],
            "framework": [
                "React",
                "Next.js",
                "Vue.js",
                "Angular",
                "Svelte",
                "Tailwind CSS",
                "Bootstrap",
                "Material UI",
                "Remix",
                "Astro",
            ],
            "tool": [
                "Webpack",
                "Vite",
                "Figma",
                "Storybook",
                "Jest",
                "Cypress",
                "Playwright",
                "Git",
                "ESLint",
                "Prettier",
            ],
            "database": ["Firebase", "Supabase", "MongoDB", "PostgreSQL", "IndexedDB"],
            "cloud": ["Vercel", "Netlify", "AWS Amplify", "Cloudflare Pages", "Render"],
            "methodology": [
                "Responsive Design",
                "Accessibility (WCAG)",
                "Agile",
                "Component-Driven Development",
                "Progressive Web Apps",
            ],
        },
        "primary_domain": "Frontend Engineering",
    },
    "data_science": {
        "titles": [
            "Data Scientist",
            "ML Engineer",
            "Research Scientist",
            "AI Engineer",
            "Data Analyst",
            "NLP Engineer",
            "LLM Engineer",
            "AI Product Engineer",
        ],
        "skills": {
            "programming_language": [
                "Python",
                "R",
                "SQL",
                "Julia",
                "Scala",
                "MATLAB",
                "Go",
            ],
            "ai_ml": [
                "PyTorch",
                "TensorFlow",
                "Scikit-learn",
                "Hugging Face",
                "LangChain",
                "LlamaIndex",
                "OpenAI API",
                "Claude API",
                "JAX",
                "Keras",
                "XGBoost",
                "LightGBM",
                "OpenCV",
            ],
            "data_tool": [
                "Pandas",
                "NumPy",
                "Polars",
                "Dask",
                "Spark",
                "Dagster",
                "Airflow",
                "MLflow",
                "DVC",
                "Kubeflow",
                "Weights & Biases",
                "Neptune",
            ],
            "data_warehouse": [
                "BigQuery",
                "Snowflake",
                "Redshift",
                "Databricks",
                "PostgreSQL",
                "MongoDB",
                "Neo4j",
                "Pinecone",
                "Qdrant",
                "ChromaDB",
                "Weaviate",
            ],
            "cloud_ml": [
                "AWS SageMaker",
                "GCP Vertex AI",
                "Azure ML",
                "Databricks",
                "Lambda Labs",
                "Hugging Face Endpoints",
                "Bedrock",
            ],
            "methodology": [
                "Statistical Modeling",
                "A/B Testing",
                "NLP",
                "Computer Vision",
                "Deep Learning",
                "MLOps",
                "RAG",
                "Fine-tuning",
                "LLM Evaluation",
                "Prompt Engineering",
                "Vector Databases",
                "Feature Engineering",
            ],
        },
        "primary_domain": "Data Science / ML / AI",
    },
    "devops": {
        "titles": [
            "DevOps Engineer",
            "SRE",
            "Cloud Engineer",
            "Infrastructure Engineer",
            "Platform Engineer",
            "Release Engineer",
            "MLOps Engineer",
        ],
        "skills": {
            "programming_language": [
                "Python",
                "Bash",
                "Go",
                "YAML",
                "HCL",
                "PowerShell",
                "Rust",
            ],
            "infrastructure": [
                "Terraform",
                "Ansible",
                "Pulumi",
                "Helm",
                "ArgoCD",
                "Crossplane",
                "CDK",
                "CloudFormation",
            ],
            "container": [
                "Docker",
                "Kubernetes",
                "K3s",
                "Podman",
                "Buildah",
                "Skopeo",
                "Istio",
                "Linkerd",
                "Envoy",
            ],
            "ci_cd": [
                "Jenkins",
                "GitHub Actions",
                "GitLab CI",
                "CircleCI",
                "Tekton",
                "Argo Workflows",
                "Azure DevOps",
            ],
            "monitoring": [
                "Prometheus",
                "Grafana",
                "ELK Stack",
                "Datadog",
                "PagerDuty",
                "Opsgenie",
                "Thanos",
                "VictoriaMetrics",
            ],
            "cloud": [
                "AWS",
                "GCP",
                "Azure",
                "DigitalOcean",
                "Linode",
                "Hetzner",
                "Oracle Cloud",
                "Alibaba Cloud",
            ],
            "ml_ops": [
                "MLflow",
                "Kubeflow",
                "Kale",
                "Seldon",
                "TensorFlow Serving",
                "TorchServe",
                "Triton Inference Server",
            ],
            "methodology": [
                "Infrastructure as Code",
                "SRE Practices",
                "Incident Management",
                "Chaos Engineering",
                "GitOps",
                "Zero Trust",
                "Platform Engineering",
                "MLOps",
            ],
        },
        "primary_domain": "DevOps / SRE / MLOps",
    },
    "fullstack": {
        "titles": [
            "Full-Stack Developer",
            "Software Engineer",
            "Web Developer",
            "Application Developer",
            "Platform Engineer",
        ],
        "skills": {
            "programming_language": [
                "JavaScript",
                "TypeScript",
                "Python",
                "Java",
                "Go",
                "Node.js",
                "C#",
            ],
            "framework": [
                "React",
                "Next.js",
                "Django",
                "FastAPI",
                "Vue.js",
                "Express.js",
                "Spring Boot",
                "Flask",
                "NestJS",
                "Angular",
                "Tailwind CSS",
            ],
            "database": [
                "PostgreSQL",
                "MongoDB",
                "Redis",
                "MySQL",
                "Supabase",
                "Firebase",
                "SQLite",
            ],
            "tool": [
                "Docker",
                "Git",
                "Kubernetes",
                "GitHub Actions",
                "Vite",
                "Webpack",
                "Storybook",
                "Jest",
                "Cypress",
            ],
            "cloud": [
                "AWS",
                "Vercel",
                "GCP",
                "Netlify",
                "Heroku",
                "Render",
                "Azure",
                "Cloudflare",
            ],
            "methodology": [
                "Agile",
                "Scrum",
                "CI/CD",
                "TDD",
                "Microservices",
                "REST API Design",
                "Responsive Design",
                "Progressive Web Apps",
            ],
        },
        "primary_domain": "Full-Stack Engineering",
    },
    "mobile": {
        "titles": [
            "Mobile Developer",
            "iOS Developer",
            "Android Developer",
            "Flutter Developer",
            "React Native Developer",
        ],
        "skills": {
            "programming_language": [
                "Swift",
                "Kotlin",
                "Dart",
                "Java",
                "Objective-C",
                "TypeScript",
                "C++",
            ],
            "framework": [
                "SwiftUI",
                "Jetpack Compose",
                "Flutter",
                "React Native",
                "Expo",
                "UIKit",
                "KMP",
            ],
            "tool": [
                "Xcode",
                "Android Studio",
                "Firebase",
                "Fastlane",
                "CocoaPods",
                "Gradle",
                "App Center",
                "TestFlight",
            ],
            "database": ["SQLite", "Realm", "Core Data", "Firebase Firestore", "Hive"],
            "cloud": ["Firebase", "AWS Amplify", "Supabase", "RevenueCat"],
            "methodology": [
                "MVVM",
                "Clean Architecture",
                "CI/CD for Mobile",
                "App Store Optimization",
                "Offline-First",
            ],
        },
        "primary_domain": "Mobile Development",
    },
    "cybersecurity": {
        "titles": [
            "Security Engineer",
            "Penetration Tester",
            "SOC Analyst",
            "Security Architect",
            "AppSec Engineer",
        ],
        "skills": {
            "programming_language": ["Python", "Bash", "C", "Go", "PowerShell", "Rust"],
            "framework": [
                "Metasploit",
                "Burp Suite",
                "OWASP ZAP",
                "Snort",
                "Suricata",
                "Sigma",
            ],
            "tool": [
                "Wireshark",
                "Nmap",
                "Nessus",
                "Splunk",
                "CrowdStrike",
                "Hashicorp Vault",
                "Trivy",
                "SonarQube",
            ],
            "database": ["PostgreSQL", "Elasticsearch", "Redis", "MongoDB"],
            "cloud": [
                "AWS Security Hub",
                "Azure Sentinel",
                "GCP Security Command Center",
            ],
            "methodology": [
                "OWASP Top 10",
                "Zero Trust Architecture",
                "Incident Response",
                "Threat Modeling",
                "SAST/DAST",
                "SOC2 Compliance",
            ],
        },
        "primary_domain": "Cybersecurity",
    },
    "product_management": {
        "titles": [
            "Product Manager",
            "Senior Product Manager",
            "Technical PM",
            "Product Owner",
            "VP of Product",
        ],
        "skills": {
            "programming_language": ["Python", "SQL", "JavaScript", "R"],
            "framework": [
                "Jira",
                "Confluence",
                "Notion",
                "Aha!",
                "Productboard",
                "Roadmunk",
            ],
            "tool": [
                "Tableau",
                "Mixpanel",
                "Amplitude",
                "Google Analytics",
                "Hotjar",
                "Figma",
                "Miro",
                "Zoom",
            ],
            "database": ["PostgreSQL", "MySQL", "BigQuery", "Snowflake"],
            "cloud": ["AWS", "GCP", "Azure"],
            "methodology": [
                "Agile",
                "Scrum",
                "Kanban",
                "OKRs",
                "Product Discovery",
                "User Research",
                "Roadmap Planning",
                "A/B Testing",
            ],
        },
        "primary_domain": "Product Management",
    },
    "design": {
        "titles": [
            "UI Designer",
            "UX Designer",
            "Product Designer",
            "Visual Designer",
            "Design Lead",
        ],
        "skills": {
            "programming_language": ["HTML", "CSS", "JavaScript", "Python"],
            "framework": [
                "Figma",
                "Sketch",
                "Adobe XD",
                "InVision",
                "Framer",
                "Principle",
            ],
            "tool": [
                "Figma",
                "Adobe Creative Suite",
                "After Effects",
                "Miro",
                "Storybook",
                "Zeplin",
                "Hotjar",
                "Maze",
            ],
            "database": ["Firebase"],
            "cloud": ["Cloudflare", "Vercel", "Netlify"],
            "methodology": [
                "User Research",
                "Design Thinking",
                "Prototyping",
                "Usability Testing",
                "Design Systems",
                "Accessibility (WCAG)",
                "Information Architecture",
            ],
        },
        "primary_domain": "Design (UI/UX)",
    },
    "qa_testing": {
        "titles": [
            "QA Engineer",
            "Software QA Engineer",
            "Test Automation Engineer",
            "SDET",
            "QA Lead",
        ],
        "skills": {
            "programming_language": ["Python", "Java", "JavaScript", "C#", "Ruby"],
            "framework": [
                "Selenium",
                "Playwright",
                "Cypress",
                "Appium",
                "JUnit",
                "TestNG",
                "PyTest",
            ],
            "tool": [
                "Jira",
                "Postman",
                "JMeter",
                "SoapUI",
                "BrowserStack",
                "Sauce Labs",
                "Docker",
                "Git",
            ],
            "database": ["PostgreSQL", "MySQL", "MongoDB", "Redis"],
            "cloud": ["AWS", "Azure", "GCP"],
            "methodology": [
                "Test Planning",
                "Test Automation",
                "Performance Testing",
                "Security Testing",
                "CI/CD Testing",
                "Agile Testing",
                "BDD",
                "TDD",
            ],
        },
        "primary_domain": "QA / Testing",
    },
    "general_it": {
        "titles": [
            "IT Specialist",
            "System Administrator",
            "IT Support Engineer",
            "Help Desk Analyst",
            "Network Administrator",
        ],
        "skills": {
            "programming_language": ["PowerShell", "Bash", "Python", "SQL", "VBA"],
            "framework": ["Active Directory", "Exchange", "SharePoint", "Office 365"],
            "tool": [
                "Azure AD",
                "Intune",
                "VMware",
                "Hyper-V",
                "SolarWinds",
                "Nagios",
                "Zabbix",
                "Docker",
                "Ansible",
            ],
            "database": ["SQL Server", "MySQL", "PostgreSQL"],
            "cloud": ["Azure", "AWS", "Google Workspace"],
            "methodology": [
                "ITIL",
                "Incident Management",
                "Change Management",
                "Help Desk",
                "System Administration",
                "Network Security",
            ],
        },
        "primary_domain": "General IT",
    },
}

UNIVERSITIES: List[Tuple[str, str]] = [
    ("MIT", "Cambridge, MA"),
    ("Stanford University", "Stanford, CA"),
    ("Carnegie Mellon University", "Pittsburgh, PA"),
    ("UC Berkeley", "Berkeley, CA"),
    ("University of Michigan", "Ann Arbor, MI"),
    ("Georgia Tech", "Atlanta, GA"),
    ("University of Toronto", "Toronto, Canada"),
    ("ETH Zurich", "Zurich, Switzerland"),
    ("Imperial College London", "London, UK"),
    ("TU Munich", "Munich, Germany"),
    ("University of Washington", "Seattle, WA"),
    ("UIUC", "Champaign, IL"),
    ("UT Austin", "Austin, TX"),
    ("Purdue University", "West Lafayette, IN"),
    ("Columbia University", "New York, NY"),
    ("University of Waterloo", "Waterloo, Canada"),
    ("NUS", "Singapore"),
    ("IIT Bombay", "Mumbai, India"),
    ("KAIST", "Daejeon, South Korea"),
    ("Technion", "Haifa, Israel"),
    ("EPFL", "Lausanne, Switzerland"),
    ("Tsinghua University", "Beijing, China"),
    ("University of Edinburgh", "Edinburgh, UK"),
    ("TU Delft", "Delft, Netherlands"),
    ("University of Melbourne", "Melbourne, Australia"),
    ("McGill University", "Montreal, Canada"),
]

CERTS: List[Tuple[str, str]] = [
    ("AWS Certified Solutions Architect", "Amazon Web Services"),
    ("AWS Certified Developer \u2013 Associate", "Amazon Web Services"),
    ("Google Cloud Professional Data Engineer", "Google"),
    ("Certified Kubernetes Administrator (CKA)", "CNCF"),
    ("Azure Solutions Architect Expert", "Microsoft"),
    ("TensorFlow Developer Certificate", "Google"),
    ("PMP", "PMI"),
    ("Scrum Master (CSM)", "Scrum Alliance"),
    ("CISSP", "ISC2"),
    ("CompTIA Security+", "CompTIA"),
    ("HashiCorp Terraform Associate", "HashiCorp"),
    ("Databricks Certified Data Engineer", "Databricks"),
    ("CKAD", "CNCF"),
    ("AWS Certified Machine Learning", "Amazon Web Services"),
    ("Google Professional Cloud Architect", "Google"),
    ("OSCP", "Offensive Security"),
    ("CSPO", "Scrum Alliance"),
    ("Professional Scrum Master (PSM)", "Scrum.org"),
    ("Certified Information Security Manager (CISM)", "ISACA"),
    ("Six Sigma Green Belt", "ASQ"),
]

HOBBIES: List[str] = [
    "Open-source contributing",
    "Technical blogging",
    "Chess",
    "Rock climbing",
    "Photography",
    "Marathon running",
    "Teaching coding workshops",
    "Board games",
    "Cooking",
    "Hiking",
    "Music production",
    "Drone building",
    "3D printing",
    "Arduino/IoT projects",
    "Podcasting",
    "Competitive programming",
    "Yoga",
    "Volunteering at local shelters",
    "Learning new languages",
    "Playing guitar",
    "Game development as hobby",
    "Reading sci-fi novels",
]

LANGUAGES: List[Tuple[str, str]] = [
    ("English", "Native"),
    ("Spanish", "Professional"),
    ("French", "Conversational"),
    ("German", "Professional"),
    ("Mandarin", "Native"),
    ("Arabic", "Native"),
    ("Hindi", "Native"),
    ("Japanese", "Basic"),
    ("Korean", "Conversational"),
    ("Portuguese", "Professional"),
    ("Russian", "Basic"),
    ("Italian", "Conversational"),
    ("Turkish", "Native"),
    ("Dutch", "Conversational"),
    ("Swedish", "Basic"),
]

COMPANIES: Dict[str, List[str]] = {
    "top": [
        "Google",
        "Meta",
        "Apple",
        "Amazon",
        "Microsoft",
        "Netflix",
        "Stripe",
        "OpenAI",
        "Anthropic",
        "Nvidia",
        "Tesla",
        "Salesforce",
    ],
    "mid": [
        "Shopify",
        "Datadog",
        "Twilio",
        "Cloudflare",
        "HashiCorp",
        "Confluent",
        "Snowflake",
        "Palantir",
        "Uber",
        "Airbnb",
        "Spotify",
        "Block",
        "Figma",
        "Notion",
        "Vercel",
        "Supabase",
    ],
    "startup": [
        "TechFlow AI",
        "CloudNova",
        "DataPulse",
        "NeuroLink Labs",
        "ScaleGrid",
        "CodeCraft Studios",
        "Quantum Edge",
        "PipelineIO",
        "DeployFast",
        "SkillBridge",
        "InfraCore",
        "ParseLab",
        "VectorDB Inc",
        "ModelShip",
        "ResumeAI",
        "CodeReview.io",
        "PromptStack",
        "AgentForge",
    ],
}

ACHIEVEMENTS: List[str] = [
    "Reduced {metric} by {pct}% through {tech} optimization",
    "Built {tech} pipeline processing {num}+ {unit} daily",
    "Led migration from {old} to {tech}, improving {metric} by {pct}%",
    "Designed {tech}-based {system} handling {num}+ concurrent {unit}",
    "Implemented {tech} integration reducing {metric} from {old_val} to {new_val}",
    "Mentored team of {team_size} engineers on {tech} best practices",
    "Automated {process} using {tech}, saving {hours}+ hours/week",
    "Delivered {feature} feature used by {num}+ {users} within {months} months",
    "Architected {tech} solution achieving {pct}% {metric} improvement",
    "Established {process} practices reducing {metric} by {pct}%",
]

PROJECTS: List[str] = [
    "AutoScale Monitor",
    "DataPipe CLI",
    "SmartCache",
    "ML Model Registry",
    "ChatBot Framework",
    "Log Aggregator",
    "API Gateway",
    "Feature Store",
    "Resume Parser",
    "Code Review Bot",
    "Deployment Dashboard",
    "Task Scheduler",
    "Real-time Analytics Engine",
    "Config Manager",
    "Health Check Service",
    "Schema Validator",
    "Token Bucket Limiter",
    "Event Sourcing Engine",
    "PDF Parser Toolkit",
    "Graph Query Builder",
    "Vector Search API",
    "Prompt Playground",
]

DOMAIN_ACHIEVEMENTS: Dict[str, List[str]] = {
    "backend": [
        "Architected and built RESTful APIs serving {num}+ requests/day using {tech}, reducing response time by {pct}%",
        "Designed and implemented microservices architecture using {tech}, handling {num}+ concurrent users",
        "Optimized database queries using {tech}, improving query performance by {pct}%",
        "Built automated CI/CD pipelines with {tech}, reducing deployment time by {pct}%",
        "Led migration from monolithic to microservices using {tech}, improving system scalability by {pct}%",
        "Implemented caching layer with {tech}, reducing database load by {pct}%",
        "Developed real-time data processing pipeline using {tech}, handling {num}+ {unit}/second",
        "Created authentication and authorization system using {tech}, securing {num}+ user accounts",
        "Built event-driven architecture using {tech}, processing {num}+ events daily",
        "Reduced infrastructure costs by {pct}% through optimization using {tech}",
        "Mentored team of {team_size} junior developers on {tech} best practices",
        "Designed and implemented message queue system using {tech}, processing {num}+ messages/hour",
    ],
    "frontend": [
        "Developed responsive web applications using {tech}, improving page load time by {pct}%",
        "Built reusable component library using {tech}, used by {num}+ developers",
        "Implemented state management solution using {tech}, reducing API calls by {pct}%",
        "Created accessible UI components using {tech}, achieving WCAG AA compliance",
        "Optimized frontend bundle size using {tech}, reducing load time by {pct}%",
        "Built real-time collaboration features using {tech}, serving {num}+ concurrent users",
        "Implemented automated testing framework using {tech}, achieving {pct}% code coverage",
        "Developed design system using {tech}, standardizing UI across {num}+ applications",
        "Created progressive web app using {tech}, increasing user engagement by {pct}%",
        "Integrated third-party APIs using {tech}, reducing integration time by {pct}%",
        "Improved website accessibility score to {pct}% using {tech}",
        "Built interactive data visualizations using {tech}, serving {num}+ daily users",
    ],
    "data_science": [
        "Developed ML models using {tech}, improving prediction accuracy by {pct}%",
        "Built end-to-end ML pipeline using {tech}, processing {num}+ samples daily",
        "Created recommendation system using {tech}, increasing user engagement by {pct}%",
        "Implemented NLP solution using {tech}, processing {num}+ text documents daily",
        "Designed data warehousing solution using {tech}, enabling analytics on {num}+ records",
        "Built real-time analytics dashboard using {tech}, used by {num}+ stakeholders",
        "Developed computer vision models using {tech}, achieving {pct}% accuracy",
        "Implemented A/B testing framework using {tech}, running {num}+ experiments",
        "Created feature engineering pipeline using {tech}, reducing preprocessing time by {pct}%",
        "Built ML model serving infrastructure using {tech}, handling {num}+ predictions/day",
        "Established MLOps practices using {tech}, reducing model deployment time by {pct}%",
        "Developed data visualization tools using {tech}, used by {num}+ analysts",
    ],
    "devops": [
        "Designed and implemented infrastructure as code using {tech}, managing {num}+ servers",
        "Built CI/CD pipelines using {tech}, reducing deployment time from {old_val} to {new_val}",
        "Implemented container orchestration using {tech}, managing {num}+ containers",
        "Created monitoring and alerting system using {tech}, reducing MTTR by {pct}%",
        "Established GitOps workflows using {tech}, improving deployment frequency by {pct}%",
        "Implemented security scanning using {tech}, identifying {num}+ vulnerabilities",
        "Built disaster recovery solution using {tech}, achieving RTO of {new_val}",
        "Optimized cloud infrastructure using {tech}, reducing costs by {pct}%",
        "Created self-service provisioning portal using {tech}, serving {num}+ engineers",
        "Implemented log aggregation using {tech}, processing {num}+ logs/day",
        "Established chaos engineering practices using {tech}, improving system resilience",
        "Built automated backup solution using {tech}, ensuring {num}% data protection",
    ],
    "fullstack": [
        "Built end-to-end web application using {tech}, serving {num}+ daily active users",
        "Architected full-stack solution with {tech}, improving response time by {pct}%",
        "Implemented server-side rendering using {tech}, reducing page load time by {pct}%",
        "Designed RESTful API and frontend using {tech}, handling {num}+ concurrent sessions",
        "Created cross-platform application using {tech}, reducing development time by {pct}%",
        "Built real-time notification system using {tech}, processing {num}+ events/day",
        "Optimized database queries and frontend rendering using {tech}, improving UX by {pct}%",
        "Led full-stack migration from {old} to {tech}, reducing latency by {pct}%",
        "Implemented CI/CD pipeline for full-stack app using {tech}, achieving {num}+ deploys/week",
        "Mentored team of {team_size} developers on {tech} full-stack best practices",
    ],
    "mobile": [
        "Developed mobile application using {tech}, achieving {num}+ downloads",
        "Built offline-first architecture using {tech}, improving app responsiveness by {pct}%",
        "Implemented push notification system using {tech}, increasing user engagement by {pct}%",
        "Created in-app purchase integration using {tech}, generating {num}+ in revenue",
        "Optimized app performance using {tech}, reducing battery consumption by {pct}%",
        "Implemented biometric authentication using {tech}, securing {num}+ user accounts",
        "Built AR/VR features using {tech}, increasing user retention by {pct}%",
        "Developed cross-platform solution using {tech}, reducing development time by {pct}%",
        "Created app analytics tracking using {tech}, providing insights on {num}+ users",
        "Implemented deep linking using {tech}, improving conversion rates by {pct}%",
        "Built widget functionality using {tech}, increasing daily active users by {pct}%",
        "Implemented app streaming using {tech}, reducing download size by {pct}%",
    ],
    "cybersecurity": [
        "Conducted penetration testing using {tech}, identifying {num}+ vulnerabilities",
        "Implemented security monitoring using {tech}, detecting {num}+ threats",
        "Built security automation using {tech}, reducing incident response time by {pct}%",
        "Designed zero-trust architecture using {tech}, securing {num}+ endpoints",
        "Implemented SIEM solution using {tech}, processing {num}+ logs/day",
        "Created vulnerability management program using {tech}, reducing CVEs by {pct}%",
        "Built secure CI/CD pipeline using {tech}, preventing {num}+ security issues",
        "Implemented identity management using {tech}, securing {num}+ user identities",
        "Developed threat intelligence platform using {tech}, tracking {num}+ threats",
        "Created security awareness training using {tech}, reducing phishing clicks by {pct}%",
        "Implemented encryption solution using {tech}, protecting {num}+ records",
        "Built incident response automation using {tech}, reducing recovery time by {pct}%",
    ],
    "product_management": [
        "Led product development using {tech}, launching features used by {num}+ users",
        "Conducted user research using {tech}, interviewing {num}+ users",
        "Implemented product analytics using {tech}, tracking {num}+ metrics",
        "Developed product roadmap using {tech}, prioritizing {num}+ features",
        "Led agile product development using {tech}, delivering {num}+ sprints",
        "Created product backlog using {tech}, managing {num}+ user stories",
        "Implemented A/B testing using {tech}, running {num}+ experiments",
        "Built customer feedback system using {tech}, collecting {num}+ insights",
        "Developed OKR tracking using {tech}, achieving {pct}% of quarterly goals",
        "Created product documentation using {tech}, serving {num}+ stakeholders",
        "Led product launches using {tech}, achieving {num}+ signups",
        "Implemented user onboarding using {tech}, improving retention by {pct}%",
    ],
    "design": [
        "Designed user interfaces using {tech}, improving usability by {pct}%",
        "Created design system using {tech}, used across {num}+ products",
        "Conducted user research using {tech}, gathering insights from {num}+ users",
        "Designed responsive layouts using {tech}, serving {num}+ screen sizes",
        "Built interactive prototypes using {tech}, reducing development cycles by {pct}%",
        "Created information architecture using {tech}, improving navigation by {pct}%",
        "Designed accessible interfaces using {tech}, achieving WCAG AA compliance",
        "Conducted usability testing using {tech}, identifying {num}+ UX issues",
        "Created brand identity using {tech}, applied across {num}+ touchpoints",
        "Designed data visualizations using {tech}, improving comprehension by {pct}%",
        "Built design collaboration workflow using {tech}, reducing review time by {pct}%",
        "Created motion design using {tech}, increasing engagement by {pct}%",
    ],
    "qa_testing": [
        "Developed test automation framework using {tech}, achieving {pct}% automation coverage",
        "Implemented CI/CD testing using {tech}, catching {num}+ defects pre-production",
        "Created performance testing suite using {tech}, identifying {num}+ bottlenecks",
        "Built security testing automation using {tech}, finding {num}+ vulnerabilities",
        "Developed API testing framework using {tech}, testing {num}+ endpoints",
        "Implemented exploratory testing using {tech}, discovering {num}+ edge cases",
        "Created test data management using {tech}, generating {num}+ test records",
        "Built mobile testing automation using {tech}, testing on {num}+ devices",
        "Implemented shift-left testing using {tech}, reducing bug escape rate by {pct}%",
        "Created test reporting dashboard using {tech}, tracking {num}+ test metrics",
        "Developed BDD framework using {tech}, writing {num}+ scenarios",
        "Built accessibility testing using {tech}, ensuring compliance for {num}+ pages",
    ],
    "general_it": [
        "Managed Windows/Active Directory infrastructure using {tech}, supporting {num}+ users",
        "Implemented IT service management using {tech}, handling {num}+ tickets/month",
        "Created network monitoring using {tech}, detecting {num}+ incidents",
        "Built automation scripts using {tech}, saving {hours}+ hours/month",
        "Implemented backup and recovery using {tech}, achieving {num}% data protection",
        "Managed cloud infrastructure using {tech}, optimizing costs by {pct}%",
        "Created IT documentation using {tech}, serving {num}+ employees",
        "Implemented identity management using {tech}, securing {num}+ accounts",
        "Built monitoring dashboards using {tech}, tracking {num}+ metrics",
        "Provided tier-2/3 support using {tech}, resolving {num}+ complex issues",
        "Implemented patch management using {tech}, patching {num}+ systems",
        "Created user training materials using {tech}, training {num}+ employees",
    ],
}

DOMAIN_PROJECTS: Dict[str, List[str]] = {
    "backend": [
        "API Gateway",
        "Microservices Platform",
        "Event Streaming Service",
        "Auth Service",
        "Payment Processing System",
        "Notification Service",
        "User Management API",
        "Search Engine",
        "Cache Layer",
        "Message Queue System",
        "File Processing Service",
        "Data Sync Platform",
        "GraphQL Server",
        "gRPC Services",
        "REST API Framework",
        "Database Proxy",
    ],
    "frontend": [
        "React Component Library",
        "Dashboard Application",
        "E-commerce Platform",
        "SPA Framework",
        "Mobile Web App",
        "Admin Panel",
        "Real-time Chat",
        "Data Visualization Tool",
        "Form Builder",
        "UI Design System",
        "Progressive Web App",
        "Content Management UI",
    ],
    "data_science": [
        "ML Model Training Pipeline",
        "Recommendation Engine",
        "NLP Text Analyzer",
        "Computer Vision System",
        "Predictive Analytics Platform",
        "Data Warehouse",
        "BI Dashboard",
        "Feature Store",
        "MLOps Platform",
        "A/B Testing Framework",
        "Data Quality Monitor",
        "Anomaly Detection System",
    ],
    "devops": [
        "Infrastructure as Code",
        "CI/CD Pipeline",
        "Kubernetes Cluster",
        "Monitoring System",
        "Log Aggregation Platform",
        "Secrets Management",
        "Auto-scaling Solution",
        "Disaster Recovery System",
        "GitOps Workflow",
        "Container Registry",
        "Observability Dashboard",
        "Chaos Engineering Framework",
    ],
    "fullstack": [
        "Task Management App",
        "E-commerce Platform",
        "Social Dashboard",
        "Real-time Chat App",
        "Project Tracker",
        "Blog Platform",
        "Analytics Dashboard",
        "Notification Hub",
        "User Management Portal",
        "Content Management System",
        "Search Application",
        "Booking System",
    ],
    "mobile": [
        "iOS Application",
        "Android Application",
        "Flutter App",
        "React Native App",
        "Mobile Backend API",
        "Push Notification Service",
        "Offline Sync",
        "In-app Purchase System",
        "App Analytics",
        "Mobile CI/CD",
        "App Store Optimization",
        "Cross-platform SDK",
    ],
    "cybersecurity": [
        "Penetration Testing Framework",
        "Security Dashboard",
        "SIEM Integration",
        "Vulnerability Scanner",
        "Security Compliance Tool",
        "Incident Response Platform",
        "Threat Intelligence Feed",
        "Security Audit System",
        "Access Control System",
        "Security Awareness Training",
        "Malware Detection",
        "Encryption Utility",
    ],
    "product_management": [
        "Product Roadmap",
        "User Research",
        "Analytics Dashboard",
        "Feature Prioritization",
        "OKR Tracking",
        "Customer Feedback System",
        "Product Backlog",
        "User Onboarding Flow",
        "A/B Testing Framework",
        "Product Documentation",
        "Release Planning",
        "Stakeholder Reporting",
    ],
    "design": [
        "Design System",
        "UI Component Library",
        "User Research",
        "Prototyping Tool",
        "Brand Identity",
        "UX Research",
        "Design Tokens",
        "Accessibility Guidelines",
        "Motion Design",
        "Data Visualization",
        "Icon Library",
        "Style Guide",
    ],
    "qa_testing": [
        "Test Automation Framework",
        "Performance Testing Suite",
        "API Testing",
        "Mobile Testing",
        "Security Testing",
        "Regression Suite",
        "Load Testing",
        "Test Data Generator",
        "BDD Framework",
        "Test Reporting",
        "CI/CD Testing",
        "Accessibility Testing",
    ],
    "general_it": [
        "IT Asset Management",
        "Help Desk System",
        "Network Monitoring",
        "Backup Solution",
        "Active Directory",
        "Patch Management",
        "User Provisioning",
        "IT Documentation",
        "Service Desk Portal",
        "Compliance Dashboard",
        "Security Policies",
        "Training Portal",
    ],
}

DOMAIN_ACTIVITIES: Dict[str, List[str]] = {
    "backend": [
        "ACM Club",
        "Open Source Contributor",
        "API Design Competition",
        "Hackathon Winner",
        "Teaching Assistant - Data Structures",
    ],
    "frontend": [
        "Google Developer Group",
        "Web Development Club",
        "UI/UX Club",
        "Accessibility Initiative",
        "Design Team Lead",
    ],
    "data_science": [
        "Research Assistant - ML Lab",
        "Data Science Club",
        "Kaggle Competition",
        "Analytics Club",
        "AI Research",
    ],
    "devops": [
        "Linux Club",
        "Infrastructure Club",
        "DevOps Bootcamp",
        "Cloud Computing Club",
        "SRE Team",
    ],
    "fullstack": [
        "Full-Stack Development Club",
        "Hackathon Team",
        "Open Source Contributor",
        "Web Dev Community",
        "Cloud-Native Club",
    ],
    "mobile": [
        "Mobile Development Club",
        "iOS Developer",
        "Android Community",
        "Flutter Community",
        "App Development Team",
    ],
    "cybersecurity": [
        "Cyber Defense Club",
        "CTF Team",
        "Security Research",
        "Ethical Hacking Club",
        "InfoSec Society",
    ],
    "product_management": [
        "Product Club",
        "Startup Club",
        "Business Analytics",
        "Entrepreneurship Club",
        "Innovation Lab",
    ],
    "design": [
        "UX Design Club",
        "Graphic Design Society",
        "Design Thinking",
        "Creative Director",
        "Art Club",
    ],
    "qa_testing": [
        "QA Club",
        "Testing Community",
        "Test Automation",
        "Quality Assurance",
        "Software Testing",
    ],
    "general_it": [
        "IT Support Club",
        "Tech Support",
        "System Administration",
        "Network Club",
        "IT Society",
    ],
}

# Domain-relevant certification pools
DOMAIN_CERT_POOL: Dict[str, List[Tuple[str, str]]] = {
    "backend": [
        ("AWS Certified Solutions Architect", "Amazon Web Services"),
        ("AWS Certified Developer Associate", "Amazon Web Services"),
        ("Google Cloud Professional Cloud Developer", "Google"),
        ("Oracle Certified Professional Java SE", "Oracle"),
        ("Certified Kubernetes Administrator", "CNCF"),
    ],
    "frontend": [
        ("Google UX Design Certificate", "Google"),
        ("Meta Front-End Developer Certificate", "Meta"),
        ("AWS Certified Developer Associate", "Amazon Web Services"),
        ("React Certified Developer", "React"),
    ],
    "data_science": [
        ("TensorFlow Developer Certificate", "Google"),
        ("AWS Certified Machine Learning", "Amazon Web Services"),
        ("Google Cloud Professional Data Engineer", "Google"),
        ("Microsoft Certified: Azure Data Scientist", "Microsoft"),
        ("Databricks Certified Data Engineer", "Databricks"),
    ],
    "devops": [
        ("Certified Kubernetes Administrator", "CNCF"),
        ("AWS Certified Solutions Architect", "Amazon Web Services"),
        ("Google Cloud Professional Cloud Engineer", "Google"),
        ("HashiCorp Terraform Associate", "HashiCorp"),
        ("Azure Administrator Associate", "Microsoft"),
    ],
    "fullstack": [
        ("AWS Certified Solutions Architect", "Amazon Web Services"),
        ("AWS Certified Developer Associate", "Amazon Web Services"),
        ("Google Cloud Professional Cloud Developer", "Google"),
        ("Certified Kubernetes Administrator", "CNCF"),
    ],
    "mobile": [
        ("Apple Certified iOS App Developer", "Apple"),
        ("Google Associate Android Developer", "Google"),
        ("AWS Certified Developer Associate", "Amazon Web Services"),
    ],
    "cybersecurity": [
        ("CISSP", "ISC2"),
        ("CompTIA Security+", "CompTIA"),
        ("CEH - Certified Ethical Hacker", "EC-Council"),
        ("AWS Certified Security Specialty", "Amazon Web Services"),
    ],
    "product_management": [
        ("Product Management Certificate", "Product School"),
        ("CSPO - Certified Scrum Product Owner", "Scrum Alliance"),
        ("PMP - Project Management Professional", "PMI"),
    ],
    "design": [
        ("Google UX Design Certificate", "Google"),
        ("Adobe Certified Expert", "Adobe"),
        ("Interaction Design Certificate", "Interaction Design Foundation"),
    ],
    "qa_testing": [
        ("ISTQB Certified Tester", "ISTQB"),
        ("Certified ScrumMaster", "Scrum Alliance"),
        ("AWS Certified DevOps Engineer", "Amazon Web Services"),
    ],
    "general_it": [
        ("CompTIA A+", "CompTIA"),
        ("CompTIA Network+", "CompTIA"),
        ("Microsoft 365 Certified: Modern Desktop Administrator", "Microsoft"),
        ("AWS Certified Cloud Practitioner", "Amazon Web Services"),
    ],
}

TIER_TEMPLATES: Dict[str, List[str]] = {
    "T1": [
        "T1_classic",
        "T1_modern",
        "T1_academic",
        "T1_functional",
        "T1_executive",
        "T1_fresher",
    ],
    "T2": [
        "T2_sidebar_left",
        "T2_sidebar_right",
        "T2_two_col",
        "T2_sidebar_icons",
        "T2_sidebar_photo",
        "T2_sidebar_timeline",
    ],
    "T3": [
        "T3_table",
        "T3_header_footer",
        "T3_nested_tables",
        "T3_card_layout",
        "T3_europass",
    ],
    "T4": ["T1_classic", "T2_sidebar_left", "T3_table", "T2_sidebar_icons"],
    "T5": ["T5_creative", "T5_minimal", "T5_infographic", "T5_magazine", "T5_dark"],
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def seniority(years: float) -> str:
    """Map years of experience to seniority level."""
    if years < 1:
        return "intern"
    if years < 2:
        return "junior"
    if years < 5:
        return "mid"
    if years < 8:
        return "senior"
    if years < 12:
        return "lead"
    if years < 18:
        return "principal"
    return "executive"


def _realistic_bullets(
    role: str,
    company: str,
    techs: List[str],
    n: int = 3,
    domain_key: str = "backend",
    sen_level: str = "mid",
) -> str:
    """Generate realistic achievement bullets based on domain and seniority."""
    templates = DOMAIN_ACHIEVEMENTS.get(domain_key, DOMAIN_ACHIEVEMENTS["backend"])
    metrics = [
        "latency",
        "response time",
        "error rate",
        "deployment time",
        "build time",
        "MTTR",
        "query time",
        "page load time",
        "CPU usage",
        "memory usage",
    ]
    units = [
        "requests",
        "users",
        "transactions",
        "events",
        "documents",
        "queries",
        "records",
        "logs",
        "messages",
    ]
    processes = [
        "CI/CD pipeline",
        "testing workflow",
        "deployment process",
        "code review",
        "monitoring",
        "incident response",
        "security scanning",
    ]

    bullets: List[str] = []
    for _ in range(n):
        tmpl = random.choice(templates)
        tech = (
            random.choice(techs)
            if techs
            else random.choice(["Python", "JavaScript", "Java", "Go"])
        )
        bullet = tmpl.format(
            tech=tech,
            metric=random.choice(metrics),
            pct=random.randint(15, 70),
            num=random.choice([100, 500, 1000, 5000, 10000, 50000, 100000]),
            unit=random.choice(units),
            old=random.choice(
                ["legacy system", "monolith", "manual process", "on-premise"]
            ),
            system=random.choice(
                ["microservice", "API", "data pipeline", "auth service", "dashboard"]
            ),
            old_val=f"{random.randint(2, 30)}s",
            new_val=f"{random.randint(50, 500)}ms",
            team_size=random.randint(2, 8),
            hours=random.randint(5, 20),
            process=random.choice(processes),
            feature=random.choice(
                ["real-time", "AI-powered", "automated", "self-service", "scalable"]
            ),
            users=random.choice(["users", "customers", "developers", "employees"]),
            months=random.randint(2, 6),
            prot=random.choice(["React", "Vue", "Angular", "Figma"]),
        )
        bullets.append(bullet)
    return "\n".join(bullets)


def _realistic_summary(
    name: str,
    domain: str,
    years: float,
    skills: List[str],
    domain_key: Optional[str] = None,
) -> str:
    """Generate realistic professional summary based on experience and domain."""
    sen = seniority(years)
    top3 = ", ".join(skills[:3]) if len(skills) >= 3 else ", ".join(skills)
    top1 = skills[0] if skills else "software development"
    top1b = skills[1] if len(skills) > 1 else "modern frameworks"
    top1c = skills[2] if len(skills) > 2 else "cloud technologies"

    y = int(years)

    if sen == "intern":
        templates = [
            f"Motivated computer science graduate with foundational skills in {top3}. Seeking to contribute to innovative projects at {domain}.",
            f"Recent graduate with coursework in {top1} and passion for learning. Ready to bring fresh perspective to {domain} challenges.",
            f"Detail-oriented intern with experience in {top1} through academic projects. Eager to develop professional skills in {domain}.",
        ]
    elif sen == "junior":
        templates = [
            f"Junior {domain.lower()} developer with {y} years of experience building applications using {top3}. Committed to writing clean, maintainable code.",
            f"Software developer specializing in {top1} with hands-on experience in {top1b}. Looking to grow in {domain}.",
            f"Junior engineer with {y} years working with {top1}. Demonstrated ability to learn quickly and contribute to team projects.",
        ]
    elif sen == "mid":
        templates = [
            f"Results-driven {domain.lower()} professional with {y} years of experience designing and implementing scalable solutions using {top3}. Track record of delivering high-impact projects.",
            f"Experienced software engineer with expertise in {top1}, {top1b}, and {top1c}. Proven ability to lead technical initiatives and mentor junior team members.",
            f"Mid-level {domain.lower()} specialist with {y} years building production systems. Skilled in {top3} with focus on performance optimization and code quality.",
        ]
    elif sen == "senior":
        templates = [
            f"Senior {domain.lower()} engineer with {y} years of experience architecting and delivering enterprise solutions. Expert in {top3} with demonstrated leadership in cross-functional teams.",
            f"Seasoned software professional with deep expertise in {top1}, {top1b}, and {top1c}. {y} years of experience delivering mission-critical systems.",
            f"Technical leader in {domain.lower()} with extensive experience designing microservices, optimizing performance, and building high-performing teams. Proficiency in {top3}.",
        ]
    elif sen == "lead":
        templates = [
            f"Lead {domain.lower()} professional with {y} years driving technical strategy and team excellence. Expert in {top3} with track record of building and scaling engineering teams.",
            f"Engineering leader with deep technical background in {top1}, {top1b}, and {top1c}. {y} years of experience delivering enterprise-scale solutions.",
            f"Senior technical leader specializing in {domain.lower()} with expertise in {top3}. Proven success in technical vision, team development, and stakeholder management.",
        ]
    else:  # principal/executive
        templates = [
            f"Principal {domain.lower()} architect with {y}+ years defining technical strategy for enterprise transformations. Subject matter expert in {top3}.",
            f"Executive-level technology leader with extensive background in {domain.lower()}. Expertise in {top3} with track record of transforming organizations through technology.",
            f"Distinguished technologist with {y} years shaping {domain.lower()} practice. Deep expertise in {top3} and proven ability to influence C-level decisions.",
        ]

    return random.choice(templates)


def _realistic_project_desc(
    name: str, techs: List[str], domain_key: str = "backend"
) -> str:
    """Generate realistic project description using actual technologies."""
    project_names = DOMAIN_PROJECTS.get(domain_key, DOMAIN_PROJECTS["backend"])
    project_name = random.choice(project_names)

    if not techs:
        techs = ["Python", "JavaScript", "React"]

    tech1, tech2 = random.sample(techs, min(2, len(techs)))

    templates = [
        f"Built {project_name} using {tech1} and {tech2}, enabling automated processing and improving efficiency by 40%",
        f"Designed and implemented {project_name} with {tech1}, integrating with {tech2} for seamless data flow",
        f"Created {project_name} leveraging {tech1} for core functionality and {tech2} for optimization",
        f"Developed {project_name} using {tech1}, reducing processing time by 60% through algorithmic improvements",
        f"Built {project_name} with {tech1} and {tech2}, serving 10,000+ daily active users",
        f"Implemented {project_name} using {tech1}, achieving 99.9% uptime and handling 50k+ requests daily",
        f"Designed {project_name} leveraging {tech1} microservices architecture with {tech2} for caching",
    ]

    return random.choice(templates)


def _realistic_certifications(domain_key: str) -> List[Tuple[str, str]]:
    """Get domain-relevant certifications."""
    domain_certs = DOMAIN_CERT_POOL.get(domain_key, DOMAIN_CERT_POOL["backend"])
    return random.sample(domain_certs, min(random.randint(1, 2), len(domain_certs)))


def _realistic_activity(domain_key: str) -> str:
    """Get domain-relevant extracurricular activity."""
    return random.choice(
        DOMAIN_ACTIVITIES.get(domain_key, DOMAIN_ACTIVITIES["backend"])
    )


# ---------------------------------------------------------------------------
# Main generators
# ---------------------------------------------------------------------------


def generate_synthetic_cv(
    cv_id: int,
    domain_key: str,
    seed: Optional[int] = None,
) -> CVSchema:
    """Generate a single synthetic CV as CVSchema v3.0.

    Args:
        cv_id: Unique CV identifier (1-based).
        domain_key: Key into DOMAINS dict (e.g. 'backend', 'frontend').
        seed: Optional seed for reproducibility.

    Returns:
        CVSchema object with all fields populated.
    """
    if seed is not None:
        random.seed(seed)
        Faker.seed(seed)

    fake = Faker()
    domain = DOMAINS[domain_key]

    # -- Identity & contact --
    name = fake.name()
    safe_name = name.lower().replace(" ", ".").replace("-", "").replace("'", "")
    email_domains = [
        "gmail.com",
        "outlook.com",
        "protonmail.com",
        "yahoo.com",
        f"{name.split()[-1].lower()}.dev",
    ]
    email = f"{safe_name}@{random.choice(email_domains)}"
    phone = fake.phone_number()
    location = f"{fake.city()}, {random.choice(['USA', 'UK', 'Canada', 'Germany', 'India', 'Singapore', 'Australia', 'UAE', 'Netherlands', 'France', 'Japan', 'Brazil'])}"
    has_github = random.random() < 0.7
    has_linkedin = random.random() < 0.85
    github_url = (
        f"https://github.com/{safe_name.replace('.', '')}" if has_github else None
    )
    linkedin_url = (
        f"https://linkedin.com/in/{name.lower().replace(' ', '-')}"
        if has_linkedin
        else None
    )
    portfolio_url = (
        f"https://{safe_name.replace('.', '')}.dev" if random.random() < 0.25 else None
    )

    contact_info = ContactInfo(
        full_name=name,
        email=email,
        phone=phone,
        location=location,
        github_url=github_url,
        linkedin_url=linkedin_url,
        portfolio_url=portfolio_url,
    )

    # -- Work experience --
    num_jobs = random.choices([1, 2, 3, 4, 5], weights=[10, 25, 35, 20, 10])[0]
    work_experience: List[WorkExperienceEntry] = []
    now = datetime.now()
    cursor = now
    all_domain_techs: List[str] = []
    for cat_skills in domain["skills"].values():
        all_domain_techs.extend(cat_skills)

    for i in range(num_jobs):
        tier_choice = random.choices(["top", "mid", "startup"], weights=[20, 45, 35])[0]
        company = random.choice(COMPANIES[tier_choice])
        dur = random.randint(6, 54)
        end = cursor
        start = end - relativedelta(months=dur)
        end_str = "Present" if i == 0 else end.strftime("%Y-%m")
        start_str = start.strftime("%Y-%m")
        role_name = random.choice(domain["titles"])
        job_techs = random.sample(
            all_domain_techs, min(random.randint(3, 6), len(all_domain_techs))
        )
        emp = random.choices(
            ["full-time", "part-time", "contract", "internship", "freelance"],
            weights=[65, 5, 15, 10, 5],
        )[0]

        job_yoe = dur / 12
        job_sen = seniority(job_yoe)

        desc = _realistic_bullets(
            role_name,
            company,
            job_techs,
            n=random.randint(2, 4),
            domain_key=domain_key,
            sen_level=job_sen,
        )

        entry = WorkExperienceEntry(
            company=company,
            organization=company,
            job_title=role_name,
            title=role_name,
            start_date=start_str,
            end_date=end_str,
            duration=f"{start_str} \u2013 {end_str}",
            duration_months=dur,
            description=desc,
            technologies=job_techs,
            is_remote=random.random() < 0.4,
            employment_type=emp,
        )
        work_experience.append(entry)
        cursor = start - relativedelta(months=random.randint(0, 6))

    total_months = sum(e.duration_months or 0 for e in work_experience)
    yoe = round(total_months / 12, 1)
    sen = seniority(yoe)

    # -- Education --
    education: List[EducationEntry] = []
    for d in range(random.choices([1, 2], weights=[55, 45])[0]):
        uni, _ = random.choice(UNIVERSITIES)
        deg = random.choice(
            [
                "B.Sc.",
                "B.Eng.",
                "M.Sc.",
                "M.Eng.",
                "Ph.D.",
                "B.A.",
                "B.Tech.",
                "M.Tech.",
            ]
        )
        field = random.choice(
            [
                "Computer Science",
                "Software Engineering",
                "Data Science",
                "Information Technology",
                "Electrical Engineering",
                "Mathematics",
                "Artificial Intelligence",
                "Cybersecurity",
            ]
        )
        grad = (now - relativedelta(years=int(yoe) + d + random.randint(0, 3))).year
        gpa = f"{random.uniform(2.8, 4.0):.2f}" if random.random() < 0.45 else None
        if d == 0 and random.random() < 0.5:
            act = _realistic_activity(domain_key)
        else:
            act = random.choice(
                [
                    "Dean's List",
                    "Teaching Assistant",
                    "Research Assistant",
                    None,
                    None,
                    None,
                ]
            )
        education.append(
            EducationEntry(
                institution=uni,
                university=uni,
                degree=deg,
                field_of_study=field,
                graduation_date=str(grad),
                gpa=gpa,
                activities=act,
            )
        )

    # -- Skills --
    skills: List[SkillEntry] = []
    for cat, skill_list in domain["skills"].items():
        for s in random.sample(skill_list, min(random.randint(2, 5), len(skill_list))):
            skills.append(SkillEntry(skill_name=s, category=cat, source="explicit"))

    explicit_names = {s.skill_name for s in skills}
    inferred: set = set()
    for e in work_experience:
        for t in e.technologies:
            if t not in explicit_names:
                inferred.add(t)
    for t in list(inferred)[: random.randint(1, 3)]:
        skills.append(SkillEntry(skill_name=t, category="other", source="inferred"))

    # -- Projects (3-5 per CV with synthetic GitHub URLs) --
    projects: List[ProjectEntry] = []
    num_projects = random.randint(3, 5)
    project_names = DOMAIN_PROJECTS.get(domain_key, DOMAIN_PROJECTS["backend"])
    selected_projects = random.sample(
        project_names, min(num_projects, len(project_names))
    )
    for idx, pname in enumerate(selected_projects):
        skill_names = [
            s.skill_name
            for s in skills
            if s.category in ("framework", "programming_language")
        ]
        if len(skill_names) >= 3:
            ptechs = random.sample(skill_names, 3)
        else:
            deficit = 3 - len(skill_names)
            extras = [s.skill_name for s in skills if s.skill_name not in skill_names]
            ptechs = skill_names + random.sample(extras, min(deficit, len(extras)))
        proj_url = f"https://github.com/synthetic-{domain_key}-{cv_id}-{idx}"
        projects.append(
            ProjectEntry(
                name=pname,
                description=_realistic_project_desc(pname, ptechs, domain_key),
                technologies=ptechs,
                url=proj_url,
                date=str(random.randint(2019, 2025)),
            )
        )

    # -- Certifications --
    certifications: List[CertificationEntry] = []
    if random.random() < 0.45:
        domain_certs = _realistic_certifications(domain_key)
        for cn, ci in domain_certs:
            if random.random() < 0.5:
                certifications.append(
                    CertificationEntry(
                        name=cn,
                        issuer=ci,
                        date=str(random.randint(2019, 2025)),
                        credential_id=fake.bothify("??##??##").upper()
                        if random.random() < 0.4
                        else None,
                    )
                )

    # -- Misc data --
    misc_data: List[MiscItem] = []
    n_misc = random.randint(1, 3)
    generators = [
        lambda: MiscItem(label="hobby", raw_text=random.choice(HOBBIES)),
        lambda: MiscItem(
            label="spoken_language",
            raw_text=f"{(l := random.choice(LANGUAGES))[0]} ({l[1]})",
        ),
        lambda: MiscItem(
            label="reference", raw_text="References available upon request"
        ),
        lambda: MiscItem(
            label="volunteering",
            raw_text=f"Volunteer coding instructor at {fake.company()} Youth Program, {random.randint(2020, 2024)}-Present",
        ),
        lambda: MiscItem(
            label="award",
            raw_text=f"{random.choice(['Best Paper Award', 'Hackathon 1st Place', 'Employee of the Quarter', 'Innovation Award', 'Outstanding Contributor'])} \u2014 {random.choice(['IEEE', 'ACM', 'Company Internal', 'Google DevFest', 'MLH'])} {random.randint(2019, 2025)}",
        ),
        lambda: MiscItem(
            label="publication",
            raw_text=f'"{fake.sentence(nb_words=8).rstrip(".")}" \u2014 {random.choice(["IEEE", "ACM", "NeurIPS", "ICML", "ArXiv", "EMNLP", "CVPR"])} {random.randint(2020, 2025)}',
        ),
        lambda: MiscItem(
            label="personal_statement",
            raw_text=f"Passionate about {random.choice(['open source', 'developer tools', 'AI ethics', 'accessibility', 'mentoring junior engineers'])} and building technology that makes a difference.",
        ),
    ]
    for _ in range(n_misc):
        misc_data.append(random.choice(generators)())

    # -- Summary --
    top_skills = [s.skill_name for s in skills[:5]]
    summary = _realistic_summary(
        name, domain["primary_domain"], yoe, top_skills, domain_key=domain_key
    )

    # -- Derived flat fields --
    skills_flat = [s.skill_name for s in skills if s.source == "explicit"]
    companies = [w.company for w in work_experience if w.company]
    job_titles = [w.job_title for w in work_experience if w.job_title]
    universities = [e.institution for e in education if e.institution]
    degrees = [e.degree for e in education if e.degree]

    # -- Tier selection --
    tier_choices = list(TIER_TEMPLATES.keys())
    tier = random.choice(tier_choices)

    return CVSchema(
        full_name=name,
        email=email,
        location=location,
        years_of_experience=yoe,
        seniority_level=sen,
        primary_domain=domain["primary_domain"],
        has_github=has_github,
        has_linkedin=has_linkedin,
        contact_info=contact_info,
        summary=summary,
        work_experience=work_experience,
        education=education,
        skills=skills,
        projects=projects,
        certifications=certifications,
        misc_data=misc_data,
        skills_flat=skills_flat,
        companies=companies,
        job_titles=job_titles,
        universities=universities,
        degrees=degrees,
        parsing_metadata={
            "cv_id": cv_id,
            "tier": tier,
            "domain": domain_key,
            "generated_at": datetime.now().isoformat(),
            "content_source": "template",
            "generator": "cv_generator_batch2",
        },
    )


def generate_batch(
    count: int = 100,
    output_dir: Optional[Path] = None,
    domain_distribution: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """Generate a batch of synthetic CVs with domain distribution.

    Args:
        count: Total number of CVs to generate.
        output_dir: If provided, save each CV as JSON to this directory.
        domain_distribution: Dict mapping domain_key -> count.
            Defaults to: backend 20, frontend 20, data_science 15,
            devops 15, fullstack 10, mobile 10, security 5,
            product_management 5.

    Returns:
        List of CV dicts (result of model_dump() on each CVSchema).
    """
    if domain_distribution is None:
        domain_distribution = {
            "backend": 20,
            "frontend": 20,
            "data_science": 15,
            "devops": 15,
            "fullstack": 10,
            "mobile": 10,
            "security": 5,
            "product_management": 5,
        }

    # Map "security" alias → "cybersecurity"
    resolved: Dict[str, int] = {}
    for key, val in domain_distribution.items():
        resolved_key = "cybersecurity" if key == "security" else key
        resolved[resolved_key] = resolved.get(resolved_key, 0) + val

    # If count doesn't match distribution total, scale proportionally
    dist_total = sum(resolved.values())
    if dist_total != count and dist_total > 0:
        scale = count / dist_total
        resolved = {k: max(1, round(v * scale)) for k, v in resolved.items()}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    cv_id = 1

    for domain_key, n in resolved.items():
        if domain_key not in DOMAINS:
            print(f"⚠ Unknown domain '{domain_key}', skipping")
            continue
        for _ in range(n):
            cv = generate_synthetic_cv(cv_id=cv_id, domain_key=domain_key)
            cv_dict = cv.model_dump()

            if output_dir is not None:
                filename = f"cv_{cv_id:05d}.json"
                filepath = output_dir / filename
                filepath.write_text(
                    json.dumps(cv_dict, indent=2, default=str), encoding="utf-8"
                )

            results.append(cv_dict)
            cv_id += 1

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        out = Path(sys.argv[3]) if len(sys.argv) > 3 else None
        batch = generate_batch(count=n, output_dir=out)
        print(f"✓ Generated {len(batch)} CVs")
    else:
        cv = generate_synthetic_cv(1, "backend")
        print(
            f"✓ Single CV: {cv.full_name}, {cv.primary_domain}, {len(cv.skills)} skills, {len(cv.projects)} projects"
        )
        for p in cv.projects:
            if p.url:
                print(f"  PROJECT_URL: {p.url}")
