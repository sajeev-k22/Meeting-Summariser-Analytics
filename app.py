from flask import Flask, request, jsonify, render_template, send_file
import os
import whisper
import dotenv
import google.generativeai as genai
import logging
from datetime import datetime, timedelta
import json
from werkzeug.utils import secure_filename
import tempfile
import shutil
import io
from collections import Counter
import re
import base64
import uuid
from threading import Lock

# Document generation imports
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

try:
    from docx import Document
    from docx.shared import Inches
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    logging.warning("python-docx not available. DOCX export will be disabled.")

# Visualization imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    import numpy as np
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logging.warning("Visualization libraries not available. Charts will be disabled.")

# Manually append the ffmpeg path so Python can access it
os.environ["PATH"] += os.pathsep + r"C:\Users\Athithya\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin"

# Load .env file
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('temp', exist_ok=True)

# Configure Google Generative AI
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
else:
    logger.warning("Google API key not found. Summarization will be disabled.")

# Initialize Whisper model
try:
    whisper_model = whisper.load_model("base")
    logger.info("Whisper model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    whisper_model = None

class MeetingAnalyzer:
    def __init__(self):
        self.allowed_extensions = {'mp3', 'mp4', 'wav', 'm4a', 'flac', 'aac', 'ogg', 'wma'}
    
    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.allowed_extensions
    
    def transcribe_audio(self, file_path):
        """Transcribe audio file using Whisper"""
        if not whisper_model:
            return {"error": "Whisper model not available"}
        
        try:
            logger.info(f"Transcribing audio file: {file_path}")
            result = whisper_model.transcribe(file_path)
            return {
                "text": result["text"],
                "language": result.get("language", "unknown"),
                "segments": result.get("segments", [])
            }
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {"error": str(e)}
    
    def summarize_meeting(self, transcript, meeting_type="general"):
        """Generate meeting summary using Google Generative AI"""
        if not GOOGLE_API_KEY:
            return {"error": "Google API key not configured"}
        
        try:
            prompt = f"""
            Please provide a comprehensive summary of this meeting transcript. 
            Meeting Type: {meeting_type}
            
            Please structure your response as follows:
            
            ## Meeting Summary
            [Brief overview of the meeting]
            
            ## Key Discussion Points
            [Main topics discussed with bullet points]
            
            ## Action Items
            [Tasks assigned with responsible parties if mentioned]
            
            ## Decisions Made
            [Important decisions reached during the meeting]
            
            ## Next Steps
            [Follow-up actions and next meeting considerations]
            
            ## Attendees (if mentioned)
            [List of people who participated]
            
            Transcript:
            {transcript}
            """
            
            response = model.generate_content(prompt)
            return {"summary": response.text}
            
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return {"error": str(e)}
    
    def analyze_sentiment(self, text):
        """Basic sentiment analysis"""
        positive_words = ['good', 'great', 'excellent', 'positive', 'successful', 'achieved', 'progress', 'improve']
        negative_words = ['bad', 'poor', 'failed', 'problem', 'issue', 'concern', 'negative', 'decline']
        
        words = re.findall(r'\b\w+\b', text.lower())
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total = positive_count + negative_count
        if total == 0:
            return "neutral"
        elif positive_count > negative_count:
            return "positive"
        else:
            return "negative"
    
    def extract_keywords(self, text, top_n=20):
        """Extract most frequent keywords"""
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        words = re.findall(r'\b\w+\b', text.lower())
        filtered_words = [word for word in words if len(word) > 3 and word not in stop_words]
        
        word_counts = Counter(filtered_words)
        return word_counts.most_common(top_n)
    
    def calculate_meeting_metrics(self, transcript, segments):
        """Calculate various meeting metrics"""
        # Determine the total duration while being tolerant to incomplete
        # or unsorted segment data returned by the transcription service.
        def _is_number(value):
            return isinstance(value, (int, float))

        total_duration = 0.0
        if segments:
            total_duration = max(
                (
                    float(segment.get('end'))
                    for segment in segments
                    if (
                        isinstance(segment, dict)
                        and _is_number(segment.get('end'))
                        and _is_number(segment.get('start'))
                    )
                ),
                default=0.0,
            )

        word_count = len(transcript.split())

        # Calculate speaking time per speaker (if available)
        speaker_stats = {}
        for segment in segments or []:
            if not isinstance(segment, dict):
                continue

            speaker = segment.get('speaker', 'Unknown')
            start_time = segment.get('start')
            end_time = segment.get('end')

            if not (_is_number(start_time) and _is_number(end_time)):
                continue

            duration = max(end_time - start_time, 0)
            speaker_stats[speaker] = speaker_stats.get(speaker, 0) + duration
        
        return {
            "duration": total_duration,
            "word_count": word_count,
            "speaker_stats": speaker_stats,
            "sentiment": self.analyze_sentiment(transcript),
            "keywords": self.extract_keywords(transcript)
        }

# Initialize analyzer
analyzer = MeetingAnalyzer()

# In-memory store for live transcription sessions
live_sessions = {}
live_sessions_lock = Lock()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not analyzer.allowed_file(file.filename):
        return jsonify({"error": "File type not supported"}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Transcribe audio
        transcription_result = analyzer.transcribe_audio(file_path)
        if "error" in transcription_result:
            return jsonify(transcription_result), 500
        
        # Log the transcription result
        logger.info(f"Transcription completed for {filename}")
        
        return jsonify({
            "success": True,
            "filename": filename,
            "transcription": transcription_result
        })
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_meeting():
    """Analyze meeting transcript"""
    try:
        data = request.get_json()
        transcript = data.get('transcript', '')
        meeting_type = data.get('meeting_type', 'general')
        
        if not transcript:
            return jsonify({"error": "No transcript provided"}), 400
        
        # Generate summary
        summary_result = analyzer.summarize_meeting(transcript, meeting_type)
        
        # Calculate metrics
        metrics = analyzer.calculate_meeting_metrics(transcript, [])
        
        return jsonify({
            "success": True,
            "summary": summary_result.get("summary", ""),
            "metrics": metrics,
            "error": summary_result.get("error")
        })
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/stream/start', methods=['POST'])
def start_stream():
    """Initialize a live audio streaming session"""
    if not whisper_model:
        return jsonify({"error": "Whisper model not available"}), 500

    session_id = uuid.uuid4().hex
    file_path = os.path.join('temp', f"live_{session_id}.webm")

    with live_sessions_lock:
        live_sessions[session_id] = {
            "file_path": file_path,
            "transcript": "",
            "metrics": {},
            "created_at": datetime.now()
        }

    return jsonify({"success": True, "session_id": session_id})

@app.route('/stream/chunk', methods=['POST'])
def stream_chunk():
    """Accept audio chunks for a live session and return incremental analytics"""
    if not whisper_model:
        return jsonify({"error": "Whisper model not available"}), 500

    data = request.get_json()
    session_id = data.get('session_id')
    chunk_b64 = data.get('audio_chunk')

    if not session_id or not chunk_b64:
        return jsonify({"error": "Missing session_id or audio_chunk"}), 400

    with live_sessions_lock:
        session = live_sessions.get(session_id)

    if not session:
        return jsonify({"error": "Session not found"}), 404

    try:
        audio_bytes = base64.b64decode(chunk_b64)
        os.makedirs(os.path.dirname(session["file_path"]), exist_ok=True)
        with open(session["file_path"], 'ab') as f:
            f.write(audio_bytes)

        # Run transcription on accumulated audio
        result = whisper_model.transcribe(session["file_path"])
        transcript_text = result.get("text", "").strip()
        segments = result.get("segments", [])

        metrics = analyzer.calculate_meeting_metrics(transcript_text, segments)

        with live_sessions_lock:
            session["transcript"] = transcript_text
            session["metrics"] = metrics

        return jsonify({
            "success": True,
            "transcript": transcript_text,
            "metrics": metrics
        })

    except Exception as e:
        logger.error(f"Streaming chunk error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/stream/stop', methods=['POST'])
def stop_stream():
    """Finalize a live audio streaming session and return full analytics"""
    data = request.get_json()
    session_id = data.get('session_id')
    meeting_type = data.get('meeting_type', 'general')

    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400

    with live_sessions_lock:
        session = live_sessions.get(session_id)

    if not session:
        return jsonify({"error": "Session not found"}), 404

    if not whisper_model:
        return jsonify({"error": "Whisper model not available"}), 500

    try:
        result = whisper_model.transcribe(session["file_path"])
        transcript_text = result.get("text", "").strip()
        segments = result.get("segments", [])

        summary_result = analyzer.summarize_meeting(transcript_text, meeting_type)
        metrics = analyzer.calculate_meeting_metrics(transcript_text, segments)

        # Clean up session
        try:
            os.remove(session["file_path"])
        except OSError:
            pass

        with live_sessions_lock:
            live_sessions.pop(session_id, None)

        return jsonify({
            "success": True,
            "transcript": transcript_text,
            "summary": summary_result.get("summary", ""),
            "metrics": metrics,
            "error": summary_result.get("error")
        })

    except Exception as e:
        logger.error(f"Streaming finalize error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate_report', methods=['POST'])
def generate_report():
    """Generate PDF report"""
    try:
        data = request.get_json()
        filename = data.get('filename', '')
        transcript = data.get('transcript', '')
        summary = data.get('summary', '')
        metrics = data.get('metrics', {})
        
        # Create temporary file for PDF
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file.close()
        
        # Generate PDF
        doc = SimpleDocTemplate(temp_file.name, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        story.append(Paragraph("Meeting Summary Report", title_style))
        story.append(Spacer(1, 12))
        
        # Meeting info
        story.append(Paragraph(f"<b>File:</b> {filename}", styles['Normal']))
        story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Summary
        story.append(Paragraph("Summary", styles['Heading2']))
        story.append(Paragraph(summary, styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Metrics
        story.append(Paragraph("Meeting Metrics", styles['Heading2']))
        metrics_data = [
            ["Metric", "Value"],
            ["Duration", f"{metrics.get('duration', 0):.1f} seconds"],
            ["Word Count", str(metrics.get('word_count', 0))],
            ["Sentiment", metrics.get('sentiment', 'unknown').title()],
        ]
        
        metrics_table = Table(metrics_data)
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(metrics_table)
        story.append(Spacer(1, 12))
        
        # Keywords
        keywords = metrics.get('keywords', [])
        if keywords:
            story.append(Paragraph("Top Keywords", styles['Heading2']))
            keyword_text = ", ".join([f"{word} ({count})" for word, count in keywords[:10]])
            story.append(Paragraph(keyword_text, styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        return send_file(temp_file.name, as_attachment=True, download_name=f"meeting_report_{filename}.pdf")
        
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    """Generate word cloud visualization"""
    if not VISUALIZATION_AVAILABLE:
        return jsonify({"error": "Visualization libraries not available"}), 500
    
    try:
        data = request.get_json()
        transcript = data.get('transcript', '')
        
        if not transcript:
            return jsonify({"error": "No transcript provided"}), 400
        
        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(transcript)
        
        # Create plot
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        
        # Save to bytes
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
        img_buffer.seek(0)
        plt.close()
        
        # Convert to base64
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        return jsonify({"wordcloud": img_base64})
        
    except Exception as e:
        logger.error(f"Word cloud generation error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/cleanup', methods=['POST'])
def cleanup_files():
    """Clean up temporary files"""
    try:
        # Clean up uploads older than 1 hour
        current_time = datetime.now()
        upload_dir = app.config['UPLOAD_FOLDER']
        
        for filename in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, filename)
            if os.path.isfile(file_path):
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                if current_time - file_time > timedelta(hours=1):
                    os.remove(file_path)
                    logger.info(f"Cleaned up old file: {filename}")

        # Clean up stale live session files
        with live_sessions_lock:
            stale_sessions = [
                (sid, session['file_path'])
                for sid, session in live_sessions.items()
                if current_time - session['created_at'] > timedelta(hours=1)
            ]
            for sid, _ in stale_sessions:
                live_sessions.pop(sid, None)

        for _, file_path in stale_sessions:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except OSError:
                pass
        
        return jsonify({"success": True})
        
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
