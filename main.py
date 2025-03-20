from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import os
from crewai import Agent, Task, Crew
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    SerperDevTool,
    WebsiteSearchTool,
    DallETool,
)
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# السماح لـ Next.js بالاتصال بـ FastAPI عبر CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # يمكنك تحديد الدومين لاحقًا لمزيد من الأمان
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# إعداد CrewAI API Keys
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# تعريف الأدوات
docs_tool = DirectoryReadTool(directory="./blog-posts")
file_tool = FileReadTool()
search_tool = SerperDevTool()
web_rag_tool = WebsiteSearchTool()
design_tool = DallETool()

# تعريف العملاء
researcher = Agent(
    role="Market Research Analyst",
    goal="Provide up-to-date market analysis of the AI industry",
    backstory="An expert analyst with a keen eye for market trends.",
    tools=[search_tool, web_rag_tool],
    verbose=True,
)

writer = Agent(
    role="Content Writer",
    goal="Craft engaging blog posts about the AI industry",
    backstory="A skilled writer with a passion for technology.",
    tools=[docs_tool, file_tool],
    verbose=True,
)

designer = Agent(
    role="Graphic Designer",
    goal="Create visually appealing and innovative designs for digital and print media",
    backstory="An innovative designer with a strong aesthetic sense.",
    tools=[design_tool, search_tool],
    verbose=True,
)

# تعريف المهام
task_status = {}


class TaskRequest(BaseModel):
    task: str  # 'research', 'write', 'design' أو 'all'


@app.get("/health-check")
def health_check():
    """تأكد من أن السيرفر يعمل"""
    return {"status": "FastAPI is running!"}


@app.post("/run-tasks")
async def run_tasks(request: TaskRequest):
    """تشغيل مهمة معينة وإعادتها إلى Next.js"""
    task = request.task.lower()

    if task not in ["research", "write", "design", "all"]:
        raise HTTPException(status_code=400, detail="Invalid task type")

    task_status[task] = "Running..."

    # تشغيل CrewAI بناءً على المهمة المحددة
    if task == "all":
        crew = Crew(
            agents=[researcher, writer, designer], tasks=[], verbose=True, planning=True
        )
    elif task == "research":
        crew = Crew(agents=[researcher], tasks=[], verbose=True, planning=True)
    elif task == "write":
        crew = Crew(agents=[writer], tasks=[], verbose=True, planning=True)
    elif task == "design":
        crew = Crew(agents=[designer], tasks=[], verbose=True, planning=True)

    await asyncio.sleep(3)  # محاكاة وقت التنفيذ
    crew.kickoff()

    task_status[task] = "Completed"

    return {"message": f"Task {task} executed successfully!"}


@app.get("/task-status")
def get_task_status():
    """إرجاع حالة جميع المهام"""
    return task_status


@app.get("/task-result/{task}")
def get_task_result(task: str):
    """جلب نتيجة تنفيذ مهمة معينة"""
    if task not in task_status:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"task": task, "status": task_status.get(task, "Unknown")}
