#!/usr/bin/env python3
"""
Script to add sample documents to Qdrant for testing the Streamlit app.
Run this after starting Qdrant to populate it with test data.
"""

import os
import sys
from dotenv import load_dotenv

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from qdrant_utils import add_document_to_qdrant
    print("✅ Qdrant utilities imported successfully")
except ImportError as e:
    print(f"❌ Failed to import Qdrant utilities: {e}")
    print("Please ensure qdrant_utils.py exists and dependencies are installed")
    sys.exit(1)

def add_sample_documents():
    """Add sample documents to Qdrant for testing"""
    
    # Sample documents with different topics
    sample_docs = [
        {
            "id": 1,
            "content": """
            Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding.
            
            Machine learning is a subset of AI that focuses on the development of algorithms and statistical models that enable computers to improve their performance on a specific task through experience. Deep learning, a subset of machine learning, uses neural networks with multiple layers to model and understand complex patterns.
            
            AI applications are widespread across various industries, including healthcare, finance, transportation, and entertainment. In healthcare, AI is used for medical diagnosis, drug discovery, and personalized treatment plans. In finance, AI algorithms analyze market trends and manage investment portfolios.
            
            The future of AI holds immense potential, with ongoing research in areas like natural language processing, computer vision, and autonomous systems. However, it also raises important ethical considerations regarding privacy, bias, and the impact on employment.
            """,
            "metadata": {"category": "technology", "topic": "Artificial Intelligence", "length": "long"}
        },
        {
            "id": 2,
            "content": """
            Climate change refers to long-term shifts in global weather patterns and average temperatures. The primary driver of current climate change is the increase in greenhouse gas concentrations in the atmosphere, largely due to human activities such as burning fossil fuels, deforestation, and industrial processes.
            
            The main greenhouse gases include carbon dioxide (CO2), methane (CH4), and nitrous oxide (N2O). These gases trap heat in the Earth's atmosphere, leading to a phenomenon known as the greenhouse effect. While the greenhouse effect is natural and necessary for life on Earth, human activities have significantly intensified it.
            
            The consequences of climate change are far-reaching and include rising global temperatures, melting polar ice caps, sea level rise, more frequent and severe weather events, and shifts in precipitation patterns. These changes affect ecosystems, agriculture, water resources, and human health.
            
            Addressing climate change requires a combination of mitigation strategies to reduce greenhouse gas emissions and adaptation measures to cope with the changes that are already occurring. This includes transitioning to renewable energy sources, improving energy efficiency, and developing climate-resilient infrastructure.
            """,
            "metadata": {"category": "environment", "topic": "Climate Change", "length": "long"}
        },
        {
            "id": 3,
            "content": """
            Programming languages are formal languages used to create computer programs and software applications. They provide a way for developers to communicate instructions to computers in a structured and understandable manner.
            
            There are hundreds of programming languages, each designed for specific purposes and use cases. High-level languages like Python, Java, and JavaScript are designed to be human-readable and are used for web development, data analysis, and general software development. Low-level languages like Assembly and C provide more direct control over hardware but are more complex to write and maintain.
            
            Python is particularly popular for beginners due to its simple syntax and readability. It's widely used in data science, machine learning, web development, and automation. JavaScript is the primary language for web development, enabling interactive websites and web applications.
            
            Learning to program involves understanding fundamental concepts such as variables, data types, control structures, functions, and object-oriented programming. Modern development also requires knowledge of version control systems like Git, testing frameworks, and deployment practices.
            """,
            "metadata": {"category": "programming", "topic": "Python", "length": "medium"}
        }
    ]
    
    print("🚀 Adding sample documents to Qdrant...")
    
    success_count = 0
    for doc in sample_docs:
        doc_id = doc["id"]
        content = doc["content"].strip()
        metadata = doc["metadata"]
        
        print(f"📝 Adding document {doc_id}: {metadata['topic']}")
        
        if add_document_to_qdrant(doc_id, content, metadata):
            print(f"✅ Successfully added document {doc_id}")
            success_count += 1
        else:
            print(f"❌ Failed to add document {doc_id}")
    
    print(f"\n🎉 Added {success_count}/{len(sample_docs)} sample documents to Qdrant")
    
    if success_count == len(sample_docs):
        print("✅ All sample documents added successfully!")
        print("\n💡 You can now test the search functionality in your Streamlit app:")
        print("   1. Type 'climate change' in the document pane")
        print("   2. Click 'Find Resources' in the Qdrant References tab")
        print("   3. Try other queries like 'programming language' or 'artificial intelligence'")
    else:
        print("⚠️ Some documents failed to add. Check your Qdrant connection and logs.")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Check if Qdrant is accessible
    qdrant_host = os.getenv("QDRANT_HOST", "http://localhost:6333")
    print(f"🔗 Connecting to Qdrant at: {qdrant_host}")
    
    try:
        add_sample_documents()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure Qdrant is running:")
        print("   docker run -p 6333:6333 qdrant/qdrant")
