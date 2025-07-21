# 🔒 Security Setup Guide

## ⚠️ CRITICAL: API Key Security

**NEVER commit API keys to version control!** This project has been secured to prevent accidental exposure.

## 🛡️ Setup Instructions

### 1. Environment Variables Setup

1. Copy the `.env.example` file to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your real API keys:
   ```bash
   # Google Gemini API Key
   GOOGLE_API_KEY=your_actual_google_api_key_here
   
   # OpenAI API Key  
   OPENAI_API_KEY=your_actual_openai_api_key_here
   
   # Anthropic Claude API Key
   ANTHROPIC_API_KEY=your_actual_anthropic_api_key_here
   ```

### 2. Verify Security

- ✅ `.env` is in `.gitignore`
- ✅ No hardcoded keys in source code
- ✅ Environment variables are loaded properly

### 3. Getting API Keys

#### Google Gemini API
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy and paste into `.env`

#### OpenAI API
1. Go to [OpenAI API Keys](https://platform.openai.com/api-keys)
2. Create a new secret key
3. Copy and paste into `.env`

#### Anthropic Claude API
1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Create a new API key
3. Copy and paste into `.env`

## 🚨 Security Checklist

Before sharing your project:

- [ ] All API keys removed from source code
- [ ] `.env` file is in `.gitignore`
- [ ] No sensitive data in commit history
- [ ] Environment variables are properly loaded
- [ ] Test with dummy keys first

## 🔧 Testing Without Real Keys

For testing purposes, you can use the Mock provider which doesn't require real API keys:

```python
# Test with mock provider only
framework = SRLPFramework()
result = framework.run_test_scenario("travel_planning", provider="mock")
```

## 📝 Best Practices

1. **Never hardcode secrets** in source code
2. **Use environment variables** for all sensitive data
3. **Rotate API keys** regularly
4. **Monitor API usage** for unusual activity
5. **Use least privilege** - only grant necessary permissions

## 🆘 If Keys Were Exposed

If you accidentally committed API keys:

1. **Immediately revoke** the exposed keys
2. **Generate new keys** from the provider
3. **Update your `.env`** file
4. **Consider rewriting git history** if needed

## 📞 Support

If you need help with security setup, please check the main README or create an issue.