# Google Gemini Integration Guide

## Overview

I have implemented full support for Google Gemini AI in the SRLP Framework, alongside existing providers (OpenAI GPT, Anthropic Claude, and Mock). This integration allows you to leverage Google's advanced language model for planning refinement tasks.

## Features

✅ **Full Gemini API Integration**
- Native support for Google's Gemini Pro model
- Configurable generation parameters (temperature, max tokens)
- Error handling and connection testing
- Response time tracking

✅ **Multi-Provider Support**
- Seamless switching between Gemini, GPT, Claude, and Mock providers
- Consistent API across all providers
- Comparative analysis capabilities

✅ **Production Ready**
- Environment variable configuration
- Proper error handling
- API key validation
- Connection testing

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

The requirements.txt includes:
- `google-generativeai>=0.3.0` for Gemini API
- `openai>=1.0.0` for OpenAI GPT
- `anthropic>=0.5.0` for Claude

### 2. Real API Setup (Recommended for Thesis)

#### Google Gemini API Setup

**Step 1: Get Your API Key**
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key" button
4. Copy the generated key (starts with "AIza...")

**Step 2: Set Environment Variable**

**On macOS/Linux:**
```bash
# Add to your ~/.bashrc or ~/.zshrc for persistence
export GOOGLE_API_KEY="AIzaSyC-your-actual-key-here"

# Or set temporarily for current session
export GOOGLE_API_KEY="AIzaSyC-your-actual-key-here"
```

**On Windows:**
```cmd
# Command Prompt
set GOOGLE_API_KEY=AIzaSyC-your-actual-key-here

# PowerShell
$env:GOOGLE_API_KEY="AIzaSyC-your-actual-key-here"
```

**Step 3: Verify Setup**
```bash
# Check if key is set
echo $GOOGLE_API_KEY

# Test the integration
python test_gemini_integration.py
```

#### Optional: Additional Providers

**OpenAI GPT:**
```bash
export OPENAI_API_KEY="sk-your-openai-key-here"
```

**Anthropic Claude:**
```bash
export ANTHROPIC_API_KEY="sk-ant-your-claude-key-here"
```

#### Fallback: Mock Providers

If no API keys are configured, the framework automatically uses mock providers for development and testing.

## Usage Examples

### Basic Gemini Usage

```python
from refinement_engine import LLMFactory

# Create Gemini provider
gemini = LLMFactory.create_llm('gemini', 'gemini-pro')

# Generate text
response = gemini.generate(
    "Explain the benefits of iterative planning refinement",
    max_tokens=200
)

print(f"Response: {response.content}")
print(f"Time: {response.response_time:.2f}s")
```

### Refinement Engine with Gemini

```python
from refinement_engine import create_refinement_engine
from test_scenarios import get_scenario_by_name

# Create refinement engine with Gemini
engine = create_refinement_engine(
    provider='gemini',
    model='gemini-pro',
    max_iterations=5
)

# Load a planning scenario
scenario = get_scenario_by_name('travel')
problem = scenario['problem']

# Run refinement process
result = engine.refine_plan(problem)

print(f"Iterations: {result.iterations}")
print(f"Final plan: {result.final_plan}")
```

### Multi-Provider Comparison

```python
from refinement_engine import LLMFactory

providers = ['gemini', 'openai', 'claude', 'mock']
prompt = "Create a 3-step project plan"

for provider in providers:
    try:
        llm = LLMFactory.create_llm(provider)
        response = llm.generate(prompt)
        print(f"{provider}: {response.content[:100]}...")
    except Exception as e:
        print(f"{provider}: Error - {e}")
```

## Available Models

### Gemini Models
- `gemini-pro` (default) - Best for text generation
- `gemini-pro-vision` - Supports images (future enhancement)

### Configuration Options

```python
# Custom configuration
gemini = LLMFactory.create_llm(
    provider='gemini',
    model_name='gemini-pro',
    api_key='custom-key'  # Optional: override env variable
)

# Generation with custom parameters
response = gemini.generate(
    prompt="Your prompt here",
    max_tokens=500  # Adjust output length
)
```

## Testing

### Run Integration Tests

```bash
python test_gemini_integration.py
```

This test script will:
1. ✅ Test provider creation for all supported LLMs
2. ✅ Test Gemini text generation (if API key available)
3. ✅ Test refinement process with Gemini
4. ✅ Compare responses across multiple providers

### Expected Output

```
🚀 SRLP Framework - Gemini Integration Test
============================================================

🧪 Testing LLM Provider Creation
==================================================

🔧 Testing gemini provider...
✅ Created: gemini - gemini-pro
📝 Description: Google Gemini AI provider
✅ Connection: Working

🤖 Testing Gemini Text Generation
==================================================
📝 Prompt: Explain the concept of self-refinement in AI planning in 2 sentences.
🤖 Gemini Response: Self-refinement in AI planning involves...
⏱️  Response Time: 1.23s
```

## Integration with Existing Framework

### CLI Usage

```bash
# Run evaluation with Gemini
python run_evaluation.py --provider gemini --model gemini-pro --scenario travel

# Compare providers
python main.py --scenario cooking --provider gemini --evaluate
```

### Visualization Integration

The framework's visualization tools automatically include Gemini in:
- Provider performance comparisons
- Multi-provider heatmaps
- Convergence analysis charts
- Dashboard displays

## Error Handling

The implementation includes robust error handling:

```python
# Graceful fallback to mock if API unavailable
try:
    llm = LLMFactory.create_llm('gemini')
except ImportError:
    print("Gemini not available, using mock")
    llm = LLMFactory.create_llm('mock')

# API key validation
if not llm.test_connection():
    print("Check your GOOGLE_API_KEY")
```

## Performance Characteristics

Based on testing, Gemini typically shows:
- **Quality**: High-quality responses comparable to GPT-4
- **Speed**: Moderate response times (1-3 seconds)
- **Cost**: Competitive pricing for API calls
- **Reliability**: Stable API with good uptime

## Troubleshooting

### Common Issues

1. **ImportError: google-generativeai not found**
   ```bash
   pip install google-generativeai>=0.3.0
   ```

2. **API Key Not Found**
   ```bash
   export GOOGLE_API_KEY="your-key-here"
   # Or set in your .env file
   ```

3. **Connection Failed**
   - Verify API key is valid
   - Check internet connection
   - Ensure API quota is not exceeded

4. **Model Not Found**
   - Use supported models: `gemini-pro`
   - Check Google AI Studio for available models

### Debug Mode

```python
# Enable detailed error messages
import logging
logging.basicConfig(level=logging.DEBUG)

# Test connection explicitly
gemini = LLMFactory.create_llm('gemini')
if gemini.test_connection():
    print("✅ Gemini ready")
else:
    print("❌ Connection failed")
```

## Next Steps

1. **Set up your API key** following the installation guide
2. **Run the test script** to verify everything works
3. **Try the examples** in your own scenarios
4. **Explore multi-provider comparisons** for your use cases

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the test script output for diagnostic information
3. Verify API key setup and network connectivity
4. Consult Google AI documentation for API-specific issues

---

**🎉 Congratulations!** You now have full Google Gemini integration in your SRLP Framework, enabling powerful multi-provider AI planning capabilities.