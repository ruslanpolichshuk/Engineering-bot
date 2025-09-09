# Railway Deployment Guide

This guide will help you deploy the RAG Assistant to Railway.

## Prerequisites

1. A Railway account (sign up at [railway.app](https://railway.app))
2. A GitHub account with this repository
3. An OpenAI API key

## Deployment Steps

### 1. Connect Repository to Railway

1. Log in to [Railway](https://railway.app)
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose this repository
5. Railway will automatically detect it's a Python application

### 2. Configure Environment Variables

In your Railway project dashboard:

1. Go to the "Variables" tab
2. Add the following environment variable:
   ```
   OPENAI_API_KEY=your_actual_openai_api_key_here
   ```

### 3. Deploy

1. Railway will automatically start building and deploying
2. The build process will:
   - Install Python dependencies from `requirements.txt`
   - Use the `Procfile` to start the application
   - Set up the web service

### 4. Access Your Application

1. Once deployed, Railway will provide a public URL
2. Click on the generated URL to access your RAG Assistant
3. The application will be available at: `https://your-app-name.railway.app`

## Configuration Details

### Procfile
The `Procfile` tells Railway how to start your application:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.enableCORS=false --server.headless=true --browser.gatherUsageStats=false
```

### Environment Variables
- `PORT`: Automatically set by Railway
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `PYTHON_VERSION`: Automatically detected by Railway

### File Structure
Railway will use these files for deployment:
- `requirements.txt`: Python dependencies
- `Procfile`: Application startup command
- `app.py`: Main Streamlit application
- `rag_assistant/`: Core application code

## Troubleshooting

### Common Issues

1. **Build Fails**: Check that all dependencies in `requirements.txt` are valid
2. **Application Won't Start**: Verify your `OPENAI_API_KEY` is set correctly
3. **Port Issues**: Railway automatically sets the `PORT` variable - don't override it
4. **Memory Issues**: Railway has memory limits - monitor usage in the dashboard

### Logs
- Check the "Deployments" tab in Railway for build logs
- Use the "Logs" tab to see runtime logs
- Common log locations: `/tmp/streamlit.log`

### Performance Optimization

1. **Memory Usage**: Monitor memory usage in Railway dashboard
2. **Cold Starts**: First request may be slower due to cold start
3. **Scaling**: Railway can auto-scale based on traffic

## Custom Domain (Optional)

1. Go to your project settings
2. Click "Domains"
3. Add your custom domain
4. Follow Railway's DNS instructions

## Monitoring

Railway provides built-in monitoring:
- CPU and memory usage
- Request metrics
- Error tracking
- Log aggregation

## Cost Considerations

Railway offers:
- Free tier with limited resources
- Pay-as-you-go pricing for production use
- Automatic scaling based on usage

## Security Notes

1. Never commit your `.env` file with real API keys
2. Use Railway's environment variables for sensitive data
3. The `.env.railway` file is a template - don't use it directly
4. Consider using Railway's secrets management for production

## Support

- Railway Documentation: [docs.railway.app](https://docs.railway.app)
- Railway Discord: [discord.gg/railway](https://discord.gg/railway)
- GitHub Issues: Create an issue in this repository