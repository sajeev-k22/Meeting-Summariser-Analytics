# Meeting Summarizer & Analytics App

A comprehensive Flask-based web application that uses AI to transcribe, summarize, and analyze meeting recordings. Built with OpenAI Whisper for transcription and Google Generative AI for summarization.

## Features

- **Audio Transcription**: Convert audio recordings to text using OpenAI's Whisper model
- **AI Summarization**: Generate comprehensive meeting summaries with key points, action items, and decisions
- **Analytics & Insights**: Sentiment analysis, keyword extraction, and meeting metrics
- **Visualizations**: Word clouds and charts for better insights
- **Document Generation**: Export results as PDF reports
- **Multiple Format Support**: MP3, MP4, WAV, M4A, FLAC, AAC, OGG, WMA

## Prerequisites

- Python 3.8 or higher
- FFmpeg installed on your system
- Google Generative AI API key

## Installation

1. **Clone or download the project**
   ```bash
   cd /Users/Ragav/Documents/sajeev
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install FFmpeg**
   - **macOS**: `brew install ffmpeg`
   - **Windows**: Download from https://ffmpeg.org/download.html and add to PATH
   - **Linux**: `sudo apt install ffmpeg`

5. **Set up environment variables**
   - Copy `env_example.txt` to `.env`
   - Get your Google API key from: https://makersuite.google.com/app/apikey
   - Update the `.env` file with your API key

## Configuration

### Google Generative AI Setup
1. Visit https://makersuite.google.com/app/apikey
2. Create a new API key
3. Add it to your `.env` file:
   ```
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

### FFmpeg Path (Windows Users)
If you're on Windows and have FFmpeg in a specific location, update the path in `app.py`:
```python
os.environ["PATH"] += os.pathsep + r"C:\path\to\your\ffmpeg\bin"
```

## Usage

1. **Start the application**
   ```bash
   python app.py
   ```

2. **Open your browser**
   Navigate to `http://localhost:5000`

3. **Upload a meeting recording**
   - Drag and drop an audio file or click to browse
   - Select the meeting type for better summarization
   - Wait for transcription and analysis to complete

4. **View results**
   - Review the transcription
   - Read the AI-generated summary
   - Explore analytics and insights
   - Generate PDF reports or export data

## API Endpoints

- `POST /upload` - Upload and transcribe audio files
- `POST /analyze` - Analyze meeting transcript and generate summary
- `POST /generate_report` - Generate PDF report
- `POST /generate_wordcloud` - Generate word cloud visualization
- `POST /cleanup` - Clean up temporary files

## File Structure

```
sajeev/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── env_example.txt       # Environment variables template
├── README.md            # This file
├── templates/           # HTML templates
│   ├── base.html
│   └── index.html
├── uploads/            # Uploaded files (created automatically)
└── temp/              # Temporary files (created automatically)
```

## Troubleshooting

### Common Issues

1. **Whisper model loading fails**
   - Ensure you have sufficient disk space (models are large)
   - Check internet connection for model download

2. **FFmpeg not found**
   - Ensure FFmpeg is installed and in your system PATH
   - For Windows, update the path in `app.py`

3. **Google API errors**
   - Verify your API key is correct
   - Check API quotas and billing

4. **File upload issues**
   - Ensure file size is under 500MB
   - Check file format is supported

### Performance Tips

- Use smaller Whisper models (tiny, base) for faster transcription
- Process shorter audio files for better performance
- Ensure adequate RAM for large file processing

## Dependencies

- **Flask**: Web framework
- **OpenAI Whisper**: Audio transcription
- **Google Generative AI**: Text summarization
- **ReportLab**: PDF generation
- **Matplotlib**: Data visualization
- **WordCloud**: Word cloud generation
- **python-docx**: DOCX export (optional)


