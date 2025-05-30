# Cardiology RAG System - Sample Questions for Testing

## Basic Cardiology Questions

### Heart Anatomy and Function
1. "What is the structure of the human heart?"
2. "How does blood flow through the heart?"
3. "What are the four chambers of the heart?"
4. "What is the function of heart valves?"
5. "Explain the cardiac cycle"

### Heart Disease and Conditions
6. "What is coronary artery disease?"
7. "What are the symptoms of a heart attack?"
8. "What is heart failure and what causes it?"
9. "What is arrhythmia and how is it treated?"
10. "What is high blood pressure and why is it dangerous?"

### Risk Factors and Prevention
11. "What are the main risk factors for heart disease?"
12. "How can I prevent cardiovascular disease?"
13. "What role does diet play in heart health?"
14. "How does exercise affect cardiovascular health?"
15. "What is the relationship between cholesterol and heart disease?"

### Diagnosis and Testing
16. "What is an ECG and what does it show?"
17. "What tests are used to diagnose heart disease?"
18. "What is cardiac catheterization?"
19. "How is blood pressure measured?"
20. "What is an echocardiogram?"

### Treatment and Management
21. "What medications are used to treat heart disease?"
22. "What is cardiac rehabilitation?"
23. "When is heart surgery necessary?"
24. "What is angioplasty and how does it work?"
25. "What lifestyle changes help manage heart disease?"

### Advanced Topics
26. "What is the difference between systolic and diastolic blood pressure?"
27. "How does diabetes affect cardiovascular health?"
28. "What are the warning signs of stroke?"
29. "What is the relationship between sleep and heart health?"
30. "How does stress affect the cardiovascular system?"

## Complex Clinical Scenarios

### Differential Diagnosis
31. "What are the different causes of chest pain?"
32. "How do you distinguish between different types of heart murmurs?"
33. "What are the various causes of shortness of breath?"

### Emergency Situations
34. "What are the immediate steps to take during a suspected heart attack?"
35. "How do you recognize and treat cardiac arrest?"
36. "What is cardiogenic shock and how is it managed?"

### Specific Populations
37. "How does heart disease affect women differently than men?"
38. "What are the cardiovascular considerations in elderly patients?"
39. "How does pregnancy affect the cardiovascular system?"

### Pharmacology
40. "What are ACE inhibitors and how do they work?"
41. "What are the side effects of beta-blockers?"
42. "How do statins help prevent heart disease?"

## Usage Examples:

### Command Line Usage:
```bash
# Single question
python cardiology_rag.py "What are the symptoms of heart failure?"

# Interactive mode
python cardiology_rag.py
```

### Python Script Usage:
```python
from cardiology_rag import CardiologyRAG

# Initialize the system
rag = CardiologyRAG()

# Ask a question
response = rag.query("What is coronary artery disease?")
print(response["answer"])

# Start interactive session
rag.interactive_session()
```

## Testing Checklist:

1. ✅ Basic heart anatomy questions
2. ✅ Common heart diseases
3. ✅ Treatment and prevention
4. ✅ Diagnostic procedures
5. ✅ Emergency scenarios
6. ✅ Risk factors
7. ✅ Medications
8. ✅ Lifestyle factors
9. ✅ Complex clinical scenarios
10. ✅ Population-specific questions

## Expected System Behavior:

- **Good Questions**: Detailed, accurate answers with source citations
- **Partial Information**: Honest acknowledgment of limitations
- **Off-topic Questions**: Polite redirection to cardiology topics
- **Emergency Questions**: Appropriate medical disclaimers
- **Technical Questions**: Clear explanations with medical terminology

## Quality Assessment Criteria:

1. **Accuracy**: Medical information should be correct
2. **Completeness**: Answers should be comprehensive
3. **Clarity**: Explanations should be understandable
4. **Source Attribution**: References to specific document sections
5. **Medical Disclaimers**: Appropriate warnings about seeking professional care
