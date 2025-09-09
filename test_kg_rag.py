#!/usr/bin/env python3
"""
Simple test script to validate KG-RAG implementation structure
"""

import os
import sys

def test_file_structure():
    """Test that all required files exist"""
    required_files = [
        '.env',
        'requirements.txt',
        'rag_assistant/config.py',
        'rag_assistant/knowledge_graph.py',
        'rag_assistant/kg_retriever.py',
        'rag_assistant/main.py',
        'rag_assistant/utils.py',
        'rag_assistant/self_rag.py',
        'app.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    else:
        print("✅ All required files exist")
        return True

def test_env_config():
    """Test .env file configuration"""
    if not os.path.exists('.env'):
        print("❌ .env file not found")
        return False
    
    with open('.env', 'r') as f:
        content = f.read()
    
    required_vars = [
        'OPENAI_API_KEY',
        'SELF_RAG_MODEL=gpt-5',
        'RECREATE=0',
        'USE_KNOWLEDGE_GRAPH=true'
    ]
    
    missing_vars = []
    for var in required_vars:
        if var not in content:
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        return False
    else:
        print("✅ Environment configuration looks good")
        return True

def test_requirements():
    """Test requirements.txt includes KG dependencies"""
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found")
        return False
    
    with open('requirements.txt', 'r') as f:
        content = f.read()
    
    kg_dependencies = [
        'neo4j',
        'networkx',
        'spacy',
        'transformers',
        'torch',
        'numpy',
        'pandas'
    ]
    
    missing_deps = []
    for dep in kg_dependencies:
        if dep not in content:
            missing_deps.append(dep)
    
    if missing_deps:
        print("❌ Missing knowledge graph dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        return False
    else:
        print("✅ Knowledge graph dependencies included")
        return True

def test_import_structure():
    """Test that Python files have correct import structure"""
    test_files = [
        'rag_assistant/config.py',
        'rag_assistant/knowledge_graph.py',
        'rag_assistant/kg_retriever.py',
        'rag_assistant/main.py'
    ]
    
    for file_path in test_files:
        if not os.path.exists(file_path):
            print(f"❌ Cannot test {file_path} - file not found")
            continue
            
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for basic Python syntax
        try:
            compile(content, file_path, 'exec')
            print(f"✅ {file_path} - syntax OK")
        except SyntaxError as e:
            print(f"❌ {file_path} - syntax error: {e}")
            return False
    
    return True

def main():
    """Run all tests"""
    print("🧪 Testing KG-RAG Implementation Structure")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Environment Config", test_env_config),
        ("Requirements", test_requirements),
        ("Import Structure", test_import_structure)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} test failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! KG-RAG implementation structure is valid.")
        print("\n📝 Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set your OpenAI API key in .env file")
        print("3. Run: streamlit run app.py")
        print("4. Test with sample PDF documents")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)