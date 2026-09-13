"""FastAPI server - Version 5: video upload + pipeline integration."""

import os
import sys
import shutil

from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware

from fastapi.staticfiles import StaticFiles
import random
import smtplib
from email.mime.text import MIMEText
from email_config import GMAIL_ADDRESS, GMAIL_APP_PASSWORD

from db import get_violations, init_db, update_status, insert_violation, get_violation_by_id, get_violations_for_video, verify_login, get_pending_violations, get_reviewed_violations, get_latest_video_for_user, get_violations_for_user_video, get_past_violations_for_user, create_pending_signup, verify_otp_and_create_account, username_exists, email_exists, get_account_by_email, create_password_reset, verify_reset_otp, reset_password, get_stats, get_reviewed_violations_for_user, get_pending_past_violations_for_user

# let this file import the pipeline code from ../pipeline
import importlib.util

pipeline_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline")
sys.path.insert(0, pipeline_dir)
import config

import re
VIDEO_FPS = 25

import subprocess
import json
from datetime import datetime, timedelta

def extract_video_metadata(video_path):
    """Uses ffprobe to pull the real recording start time and true
    frame rate from a video file, if the file has this metadata
    embedded (most phone-recorded videos do; older/re-exported
    dataset videos may or may not)."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
            capture_output=True, text=True, timeout=10
        )
        data = json.loads(result.stdout)
        tags = data.get("format", {}).get("tags", {})
        creation_time = tags.get("creation_time")

        fps = None
        fps_tag = tags.get("com.android.capture.fps")
        if fps_tag:
            fps = float(fps_tag)

        return creation_time, fps
    except Exception:
        return None, None


def get_video_timestamp(snapshot_path, video_start_time, video_fps):
    """Computes the actual real-world time a violation occurred,
    using the video's true recording start time + how far into the
    video the violation frame was. Falls back to a relative
    'time into video' string if real start time isn't available."""
    match = re.search(r"_frame(\d+)\.png", snapshot_path or "")
    if not match:
        return "Unknown"
    frame_num = int(match.group(1))
    fps = video_fps or VIDEO_FPS
    seconds_in = frame_num / fps

    if video_start_time:
        try:
            start = datetime.fromisoformat(video_start_time.replace("Z", "+00:00"))
            actual_time_utc = start + timedelta(seconds=seconds_in)
            actual_time_ist = actual_time_utc + timedelta(hours=5, minutes=30)
            return actual_time_ist.strftime("%d %b %Y, %I:%M:%S %p IST")
        except Exception:
            pass

    minutes = int(seconds_in // 60)
    seconds = int(seconds_in % 60)
    return f"{minutes}:{seconds:02d}"


# load pipeline/main.py under a distinct name, since it's also called "main.py"
# just like this file - avoids Python confusing the two
spec = importlib.util.spec_from_file_location("pipeline_main", os.path.join(pipeline_dir, "main.py"))
pipeline_main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline_main)
run_pipeline = pipeline_main.run_pipeline

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="change-this-secret-key-later")
init_db()

# serve snapshot images so <img> tags can load them directly
app.mount("/snapshots", StaticFiles(directory="snapshots"), name="snapshots")
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# tracks the most recently uploaded video's filename, so the dashboard
# can show only that video's results by default
current_video = {"filename": None}


def send_otp_email(to_email, otp):
    """Sends a one-time verification code to the given email address."""
    msg = MIMEText(f"Your verification code is: {otp}\n\nEnter this code to complete your signup.")
    msg["Subject"] = "Your Verification Code"
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = to_email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.send_message(msg)


@app.get("/signup", response_class=HTMLResponse)
def signup_page(error: str = None):
    error_msg = f'<p style="color:var(--danger-text)">{error}</p>' if error else ""
    return f"""
    <html>
    <head>
        <title>HCS | Sign up</title>
        <link rel="icon" type="image/svg+xml" href="/static/favicon.svg">
        <link rel="stylesheet" href="/static/style.css">
        <script src="/static/theme.js"></script>
    </head>
    <body>
        <button class="theme-toggle theme-toggle-fixed" onclick="toggleTheme()">Toggle theme</button>
        <div class="brand-header">
            <div class="logo">
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M3.5 13.5C3.5 7.7 7.2 3.5 12 3.5C16.8 3.5 20.5 7.7 20.5 13.5" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>
                    <path d="M3 13.5H21" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>
                    <path d="M5 13.5V15.5C5 16.6 5.9 17.5 7 17.5H17C18.1 17.5 19 16.6 19 15.5V13.5" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M9.5 13.5V16" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>
                </svg>
            </div>
            <h1>Create your account</h1>
            <p>Sign up to get started</p>
        </div>
        <div class="container" style="max-width:400px;">
            <div class="card auth-card">
                {error_msg}
                <form method="post" action="/signup">
                    <label class="field-label">Username</label>
                    <input type="text" name="username" placeholder="Choose a username" required>
                    <label class="field-label">Email</label>
                    <input type="email" name="email" placeholder="Enter your email" required>
                    <label class="field-label">Password</label>
                    <input type="password" name="password" placeholder="Choose a password" required>
                    <button type="submit" class="btn-primary" style="width:100%; padding:13px; margin-top:8px;">Send verification code</button>
                </form>
                <div class="auth-footer">
                    <p>Already have an account? <a href="/login">Log in</a></p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """


@app.post("/signup")
def do_signup(username: str = Form(...), email: str = Form(...), password: str = Form(...)):
    if username_exists(username):
        return RedirectResponse(url="/signup?error=Username+already+taken", status_code=303)
    if email_exists(email):
        return RedirectResponse(url="/signup?error=An+account+with+this+email+already+exists", status_code=303)

    otp = str(random.randint(100000, 999999))
    create_pending_signup(username, email, password, otp)

    try:
        send_otp_email(email, otp)
    except Exception as e:
        return RedirectResponse(url=f"/signup?error=Could+not+send+email:+{e}", status_code=303)

    return RedirectResponse(url=f"/verify-otp?username={username}", status_code=303)


@app.get("/verify-otp", response_class=HTMLResponse)
def verify_otp_page(username: str, error: str = None):
    error_msg = f'<p style="color:var(--danger-text)">Incorrect code, try again.</p>' if error else ""
    return f"""
    <html>
    <head>
        <title>HCS | Verify Email</title>
        <link rel="icon" type="image/svg+xml" href="/static/favicon.svg">
        <link rel="stylesheet" href="/static/style.css">
        <script src="/static/theme.js"></script>
    </head>
    <body>
        <button class="theme-toggle theme-toggle-fixed" onclick="toggleTheme()">Toggle theme</button>
        <div class="brand-header">
            <div class="logo">
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M3.5 13.5C3.5 7.7 7.2 3.5 12 3.5C16.8 3.5 20.5 7.7 20.5 13.5" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>
                    <path d="M3 13.5H21" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>
                    <path d="M5 13.5V15.5C5 16.6 5.9 17.5 7 17.5H17C18.1 17.5 19 16.6 19 15.5V13.5" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M9.5 13.5V16" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>
                </svg>
            </div>
            <h1>Check your email</h1>
            <p>We sent a 6-digit code to your inbox</p>
        </div>
        <div class="container" style="max-width:400px;">
            <div class="card auth-card">
                {error_msg}
                <form method="post" action="/verify-otp">
                    <input type="hidden" name="username" value="{username}">
                    <label class="field-label">Verification code</label>
                    <input type="text" name="otp" placeholder="Enter 6-digit code" required>
                    <button type="submit" class="btn-primary" style="width:100%; padding:13px; margin-top:8px;">Verify</button>
                </form>
            </div>
        </div>
    </body>
    </html>
    """


@app.post("/verify-otp")
def do_verify_otp(username: str = Form(...), otp: str = Form(...)):
    success = verify_otp_and_create_account(username, otp)
    if not success:
        return RedirectResponse(url=f"/verify-otp?username={username}&error=1", status_code=303)
    return RedirectResponse(url="/login", status_code=303)



@app.get("/login", response_class=HTMLResponse)
def login_page(error: str = None):
    error_msg = f'<p style="color:var(--danger-text)">Invalid username or password.</p>' if error else ""
    return f"""
    <html>
    <head>
        <title>HCS | Login</title>
        <link rel="icon" type="image/svg+xml" href="/static/favicon.svg">
        <link rel="stylesheet" href="/static/style.css">
        <script src="/static/theme.js"></script>
    </head>
    <body>
        <button class="theme-toggle theme-toggle-fixed" onclick="toggleTheme()">Toggle theme</button>
        <div class="brand-header">
            <div class="logo">
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M3.5 13.5C3.5 7.7 7.2 3.5 12 3.5C16.8 3.5 20.5 7.7 20.5 13.5" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>
                    <path d="M3 13.5H21" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>
                    <path d="M5 13.5V15.5C5 16.6 5.9 17.5 7 17.5H17C18.1 17.5 19 16.6 19 15.5V13.5" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M9.5 13.5V16" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>
                </svg>
            </div>
            <h1>Helmet Compliance System</h1>
            <p>Sign in to your account</p>
        </div>
        <div class="container" style="max-width:400px;">
            <div class="card auth-card">
                {error_msg}
                <form method="post" action="/login">
                    <label class="field-label">Username</label>
                    <input type="text" name="username" placeholder="Enter your username" required>
                    <label class="field-label">Password</label>
                    <input type="password" name="password" placeholder="Enter your password" required>
                    <button type="submit" class="btn-primary" style="width:100%; padding:12px; margin-top:8px;">Log in</button>
                </form>
                <div class="auth-footer">
                    <p>New here? <a href="/signup">Create an account</a></p>
                    <p><a href="/forgot-password">Forgot your password?</a></p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """


@app.post("/login")
def do_login(request: Request, username: str = Form(...), password: str = Form(...)):
    account = verify_login(username, password)
    if account is None:
        return RedirectResponse(url="/login?error=1", status_code=303)
    request.session["username"] = account["username"]
    request.session["role"] = account["role"]
    return RedirectResponse(url="/", status_code=303)
@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)



@app.get("/forgot-password", response_class=HTMLResponse)
def forgot_password_page(error: str = None):
    error_msg = f'<p style="color:var(--danger-text)">{error}</p>' if error else ""
    return f"""
    <html>
    <head>
        <title>HCS | Forgot Password</title>
        <link rel="icon" type="image/svg+xml" href="/static/favicon.svg">
        <link rel="stylesheet" href="/static/style.css">
        <script src="/static/theme.js"></script>
    </head>
    <body>
        <button class="theme-toggle theme-toggle-fixed" onclick="toggleTheme()">Toggle theme</button>
        <div class="brand-header">
            <div class="logo">
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M3.5 13.5C3.5 7.7 7.2 3.5 12 3.5C16.8 3.5 20.5 7.7 20.5 13.5" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>
                    <path d="M3 13.5H21" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>
                    <path d="M5 13.5V15.5C5 16.6 5.9 17.5 7 17.5H17C18.1 17.5 19 16.6 19 15.5V13.5" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M9.5 13.5V16" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>
                </svg>
            </div>
            <h1>Reset your password</h1>
            <p>Enter the email linked to your account</p>
        </div>
        <div class="container" style="max-width:400px;">
            <div class="card auth-card">
                {error_msg}
                <form method="post" action="/forgot-password">
                    <label class="field-label">Email</label>
                    <input type="email" name="email" placeholder="Enter your email" required>
                    <button type="submit" class="btn-primary" style="width:100%; padding:13px; margin-top:8px;">Send reset code</button>
                </form>
                <p class="muted" style="margin-top:16px; text-align:center;"><a href="/login">Back to login</a></p>
            </div>
        </div>
    </body>
    </html>
    """


@app.post("/forgot-password")
def do_forgot_password(email: str = Form(...)):
    account = get_account_by_email(email)
    if account is None:
        return RedirectResponse(url="/forgot-password?error=No+account+found+with+this+email", status_code=303)

    otp = str(random.randint(100000, 999999))
    create_password_reset(account["username"], otp)

    try:
        send_otp_email(email, otp)
    except Exception as e:
        return RedirectResponse(url=f"/forgot-password?error=Could+not+send+email:+{e}", status_code=303)

    return RedirectResponse(url=f"/reset-password?username={account['username']}", status_code=303)


@app.get("/reset-password", response_class=HTMLResponse)
def reset_password_page(username: str, error: str = None):
    error_msg = f'<p style="color:var(--danger-text)">{error}</p>' if error else ""
    return f"""
    <html>
    <head>
        <title>HCS | Reset Password</title>
        <link rel="icon" type="image/svg+xml" href="/static/favicon.svg">
        <link rel="stylesheet" href="/static/style.css">
        <script src="/static/theme.js"></script>
    </head>
    <body>
        <button class="theme-toggle theme-toggle-fixed" onclick="toggleTheme()">Toggle theme</button>
        <div class="brand-header">
            <div class="logo">
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M3.5 13.5C3.5 7.7 7.2 3.5 12 3.5C16.8 3.5 20.5 7.7 20.5 13.5" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>
                    <path d="M3 13.5H21" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>
                    <path d="M5 13.5V15.5C5 16.6 5.9 17.5 7 17.5H17C18.1 17.5 19 16.6 19 15.5V13.5" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M9.5 13.5V16" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>
                </svg>
            </div>
            <h1>Enter reset code</h1>
            <p>We sent a 6-digit code to your email</p>
        </div>
        <div class="container" style="max-width:400px;">
            <div class="card auth-card">
                {error_msg}
                <form method="post" action="/reset-password">
                    <input type="hidden" name="username" value="{username}">
                    <label class="field-label">Verification code</label>
                    <input type="text" name="otp" placeholder="6-digit code" required>
                    <label class="field-label">New password</label>
                    <input type="password" name="new_password" placeholder="Enter a new password" required>
                    <button type="submit" class="btn-primary" style="width:100%; padding:13px; margin-top:8px;">Reset password</button>
                </form>
            </div>
        </div>
    </body>
    </html>
    """


@app.post("/reset-password")
def do_reset_password(username: str = Form(...), otp: str = Form(...), new_password: str = Form(...)):
    if not verify_reset_otp(username, otp):
        return RedirectResponse(url=f"/reset-password?username={username}&error=Incorrect+code", status_code=303)

    reset_password(username, new_password)
    return RedirectResponse(url="/login", status_code=303)

    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


@app.get("/api/violations")
def api_violations():
    return get_violations()


@app.post("/approve/{violation_id}")
def approve(violation_id: int, request: Request):
    if request.session.get("role") != "admin":
        return RedirectResponse(url="/", status_code=303)
    update_status(violation_id, "approved")
    return RedirectResponse(url="/", status_code=303)


@app.post("/reject/{violation_id}")
def reject(violation_id: int, request: Request):
    if request.session.get("role") != "admin":
        return RedirectResponse(url="/", status_code=303)
    update_status(violation_id, "rejected")
    return RedirectResponse(url="/", status_code=303)


@app.post("/upload")
def upload_video(request: Request, video: UploadFile = File(...)):
    # save the uploaded video to disk
    video_path = os.path.join(UPLOAD_DIR, video.filename)
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    request.session["current_video"] = video.filename

    video_start_time, video_fps = extract_video_metadata(video_path)

    try:
        # run the actual pipeline on this video
        manager, observations, plate_matches, sample_frames = run_pipeline(
            video_path=video_path
        )
    except Exception as e:
        return HTMLResponse(f"""
        <html>
        <head>
            <title>HCS | Upload Failed</title>
        <link rel="icon" type="image/svg+xml" href="/static/favicon.svg">
            <link rel="stylesheet" href="/static/style.css">
            <script src="/static/theme.js"></script>
        </head>
        <body>
            <div class="brand-header">
                <h1>Upload failed</h1>
                <p>We couldn\'t process this video</p>
            </div>
            <div class="container" style="max-width:500px;">
                <div class="card auth-card">
                    <p style="color:var(--danger-text);">
                        This usually means the video file didn\'t upload completely,
                        or is corrupted. Please try uploading it again.
                    </p>
                    <p class="muted" style="font-size:12px; margin-top:16px;">
                        Technical details: {type(e).__name__}
                    </p>
                    <a href="/" class="btn btn-primary" style="display:block; text-align:center; margin-top:16px; text-decoration:none;">Back to dashboard</a>
                </div>
            </div>
        </body>
        </html>
        """, status_code=500)

    # write each confirmed violation into the database
    import cv2
    snapshot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)

    for rider_id, frame_num, img in sample_frames:
        inst = manager.instances_by_rider.get(rider_id)
        if inst is None:
            continue
        snapshot_filename = f"violation_rider{rider_id}_frame{frame_num}.png"
        snapshot_path = os.path.join(snapshot_dir, snapshot_filename)
        cv2.imwrite(snapshot_path, img)

        insert_violation(
            rider_id=rider_id,
            violation_type="no_helmet",
            snapshot_path=snapshot_filename,
            confidence=inst.confidence,
            video_source=video.filename,
            uploaded_by=request.session.get("username"),
            video_start_time=video_start_time,
            video_fps=video_fps,
        )

    return RedirectResponse(url="/", status_code=303)


@app.get("/violation/{violation_id}", response_class=HTMLResponse)
def violation_detail(request: Request, violation_id: int):
    username = request.session.get("username")
    role = request.session.get("role")
    if username is None:
        return RedirectResponse(url="/login", status_code=303)

    v = get_violation_by_id(violation_id)
    if v is None:
        return HTMLResponse("<h2>Violation not found</h2>", status_code=404)

    is_admin = (role == "admin")
    badge_class = {"pending": "badge-pending", "approved": "badge-approved", "rejected": "badge-rejected"}
    status_badge = f'<span class="badge {badge_class.get(v["status"], "")}">{v["status"]}</span>'
    confidence_text = f"{v['confidence']*100:.1f}%" if v['confidence'] is not None else "Not available"
    plate_text = v.get("plate_number") or "Not available (OCR not yet implemented)"

    action_buttons = f"""
        <form method="post" action="/approve/{v['id']}" style="display:inline">
            <button type="submit" class="btn-primary" onclick="return confirmAction('Approve this violation?', this.form)">Approve</button>
        </form>
        <form method="post" action="/reject/{v['id']}" style="display:inline">
            <button type="submit" class="btn-danger" onclick="return confirmAction('Reject this violation?', this.form)">Reject</button>
        </form>
    """ if (is_admin and v['status'] == 'pending') else ""

    return f"""
    <html>
    <head>
        <title>HCS | Violation #{v['id']}</title>
        <link rel="icon" type="image/svg+xml" href="/static/favicon.svg">
        <link rel="stylesheet" href="/static/style.css">
        <script src="/static/theme.js"></script>
    </head>
    <body>
        <div class="dashboard-topbar">
            <div class="brand">
                <div class="logo">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M3.5 13.5C3.5 7.7 7.2 3.5 12 3.5C16.8 3.5 20.5 7.7 20.5 13.5" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>
                        <path d="M3 13.5H21" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>
                        <path d="M5 13.5V15.5C5 16.6 5.9 17.5 7 17.5H17C18.1 17.5 19 16.6 19 15.5V13.5" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M9.5 13.5V16" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>
                    </svg>
                </div>
                Helmet Compliance System
            </div>
            <div class="user-info">
                <span class="role-badge">{role}</span>
                <span>{username}</span>
                <a href="/logout">Log out</a>
                <button class="theme-toggle" onclick="toggleTheme()"></button>
            </div>
        </div>
        <div class="dashboard-container" style="max-width:800px;">
            <p style="margin-bottom:20px;"><a href="/">&larr; Back to all violations</a></p>
            <div class="card">
                <img src="/snapshots/{v['snapshot_path']}" class="snapshot" style="margin-bottom:24px;">

                <table class="detail-table">
                    <tr><td>Violation ID</td><td>#{v['id']}</td></tr>
                    <tr><td>Rider ID</td><td>{v['rider_id']}</td></tr>
                    <tr><td>Violation Type</td><td>{v['violation_type'].replace('_', ' ').title()}</td></tr>
                    <tr><td>Violation Time</td><td>{get_video_timestamp(v['snapshot_path'], v.get('video_start_time'), v.get('video_fps'))}</td></tr>
                    <tr><td>Status</td><td>{status_badge}</td></tr>
                    <tr><td>Confidence</td><td>{confidence_text}</td></tr>
                    <tr><td>License Plate</td><td>{plate_text}</td></tr>
                    <tr><td>Location</td><td>Not available (no GPS data in source video)</td></tr>
                </table>

                {f'<div style="margin-top:24px;">{action_buttons}</div>' if action_buttons else ""}
            </div>
        </div>
        <div id="modal-overlay" class="modal-overlay">
            <div class="modal-box">
                <p id="modal-message"></p>
                <div class="modal-actions">
                    <button class="btn" onclick="modalCancel()">Cancel</button>
                    <button class="btn-primary" onclick="modalConfirm()">Confirm</button>
                </div>
            </div>
        </div>
    </body>
    </html>
    """


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    username = request.session.get("username")
    role = request.session.get("role")

    if username is None:
        return RedirectResponse(url="/login", status_code=303)

    is_admin = (role == "admin")

    badge_class = {"pending": "badge-pending", "approved": "badge-approved", "rejected": "badge-rejected"}

    def render_table(violations, show_actions, show_uploader=False):
        if not violations:
            return '<div class="empty-state">Nothing here yet.</div>'

        # group violations by the date they were detected/processed
        groups = {}
        for v in violations:
            date_key = v["detected_at"][:10]  # YYYY-MM-DD
            groups.setdefault(date_key, []).append(v)

        action_header = "<th>Action</th>" if show_actions else ""
        uploader_header = "<th>Uploaded By</th>" if show_uploader else ""

        all_sections = ""
        for group_index, date_key in enumerate(sorted(groups.keys(), reverse=True)):
            group_violations = groups[date_key]
            try:
                display_date = datetime.strptime(date_key, "%Y-%m-%d").strftime("%d %B %Y")
            except Exception:
                display_date = date_key

            group_id = date_key.replace("-", "")
            is_collapsed = group_index > 0  # first (most recent) group starts open

            rows = ""
            for v in group_violations:
                action_cell = f"""
                    <form method="post" action="/approve/{v['id']}" style="display:inline">
                        <button type="submit" class="btn-primary" style="padding:6px 12px; font-size:12px;" onclick="return confirmAction('Approve this violation?', this.form)">Approve</button>
                    </form>
                    <form method="post" action="/reject/{v['id']}" style="display:inline">
                        <button type="submit" class="btn-danger" style="padding:6px 12px; font-size:12px;" onclick="return confirmAction('Reject this violation?', this.form)">Reject</button>
                    </form>
                    """
                status_badge = f'<span class="badge {badge_class.get(v["status"], "")}">{v["status"]}</span>'
                plate = v.get("plate_number") or "-"
                action_col = f"<td>{action_cell}</td>" if show_actions else ""
                uploader_col = f"<td>{v.get('uploaded_by') or '-'}</td>" if show_uploader else ""
                rows += f"""
                <tr>
                    <td><a href="/violation/{v['id']}">#{v['id']}</a></td>
                    <td>{v['rider_id']}</td>
                    <td>{v['violation_type'].replace('_', ' ').title()}</td>
                    <td>{plate}</td>
                    {uploader_col}
                    <td>{get_video_timestamp(v['snapshot_path'], v.get('video_start_time'), v.get('video_fps'))}</td>
                    <td>{status_badge}</td>
                    {action_col}
                </tr>
                """

            collapsed_class = "collapsed" if is_collapsed else ""
            all_sections += f"""
            <div id="header-{group_id}" class="date-header {collapsed_class}" onclick="toggleDateGroup('{group_id}')">
                <span class="chevron">&#9660;</span>
                <h3 style="margin:0; font-size:15px; color:var(--text-secondary);">{display_date} ({len(group_violations)})</h3>
            </div>
            <div id="table-{group_id}" class="date-group-table {collapsed_class}">
                <table>
                    <tr>
                        <th>ID</th><th>Rider</th><th>Type</th><th>Plate</th>{uploader_header}<th>Violation Time</th><th>Status</th>{action_header}
                    </tr>
                    {rows}
                </table>
            </div>
            """
        return all_sections



    if is_admin:
        pending = get_pending_violations()
        reviewed = get_reviewed_violations()

        pending_section = render_table(pending, show_actions=True, show_uploader=True)
        stats = get_stats()
        stats_row = f"""
        <div class="stats-row">
            <div class="stat-card"><div class="stat-value">{stats['total']}</div><div class="stat-label">Total Violations</div></div>
            <div class="stat-card"><div class="stat-value">{stats['pending']}</div><div class="stat-label">Pending</div></div>
            <div class="stat-card"><div class="stat-value">{stats['approved']}</div><div class="stat-label">Approved</div></div>
            <div class="stat-card"><div class="stat-value">{stats['rejected']}</div><div class="stat-label">Rejected</div></div>
        </div>
        """
        past_section = render_table(reviewed, show_actions=False, show_uploader=True)

        body = f"""
        {stats_row}
        <div class="tabs">
            <button id="btn-pending" class="tab-btn active" onclick="showTab('pending')">Pending Review</button>
            <button id="btn-past" class="tab-btn" onclick="showTab('past')">Past Reviews</button>
        </div>
        <div id="tab-pending" class="tab-content active">
            {pending_section}
        </div>
        <div id="tab-past" class="tab-content">
            {past_section}
        </div>
        """

    else:
        latest_video = request.session.get("current_video")

        if latest_video is None:
            current_section = '<div class="empty-state">You haven\'t uploaded any video yet.</div>'
            past_uploads = get_pending_past_violations_for_user(username, "")
        else:
            current_violations = get_violations_for_user_video(username, latest_video)
            current_section = render_table(current_violations, show_actions=False)
            past_uploads = get_pending_past_violations_for_user(username, latest_video)

        past_uploads_section = render_table(past_uploads, show_actions=False)
        reviewed_violations = get_reviewed_violations_for_user(username)
        reviewed_section = render_table(reviewed_violations, show_actions=False)

        body = f"""
        <div class="card">
            <h2 style="margin-top:0;">Upload a video</h2>
            <div class="upload-options">
                <div class="upload-option-btn" onclick="triggerCameraInput()">Record Now</div>
                <div class="upload-option-btn" onclick="triggerFileInput()">Choose File</div>
            </div>
            <div id="selected-file-name" class="selected-file-name"></div>
            <form method="post" action="/upload" enctype="multipart/form-data" class="upload-form" onsubmit="event.preventDefault(); handleUploadSubmitWithCheck(this).then(ok => {{ if (ok !== false) this.submit(); }});">
                <input type="file" accept="video/*" capture="environment" id="camera-input" style="display:none" onchange="onVideoSelected(this)">
                <input type="file" accept="video/*" id="file-input" style="display:none" onchange="onVideoSelected(this)">
                <input type="file" name="video" accept="video/*" required id="video-input" style="display:none">
                <button type="submit" class="btn-primary">Upload &amp; Process</button>
            </form>
        </div>
        <div class="tabs">
            <button id="btn-current" class="tab-btn active" onclick="showTab('current')">Current Upload</button>
            <button id="btn-uploads" class="tab-btn" onclick="showTab('uploads')">Pending</button>
            <button id="btn-reviewed" class="tab-btn" onclick="showTab('reviewed')">Reviewed</button>
        </div>
        <div id="tab-current" class="tab-content active">
            {current_section}
        </div>
        <div id="tab-uploads" class="tab-content">
            {past_uploads_section}
        </div>
        <div id="tab-reviewed" class="tab-content">
            {reviewed_section}
        </div>
        """

    return f"""
    <html>
    <head>
        <title>HCS | Dashboard</title>
        <link rel="icon" type="image/svg+xml" href="/static/favicon.svg">
        <link rel="stylesheet" href="/static/style.css">
        <script src="/static/theme.js"></script>
    </head>
    <body>
        <div class="dashboard-topbar">
            <div class="brand">
                <div class="logo">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M3.5 13.5C3.5 7.7 7.2 3.5 12 3.5C16.8 3.5 20.5 7.7 20.5 13.5" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>
                        <path d="M3 13.5H21" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>
                        <path d="M5 13.5V15.5C5 16.6 5.9 17.5 7 17.5H17C18.1 17.5 19 16.6 19 15.5V13.5" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M9.5 13.5V16" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>
                    </svg>
                </div>
                Helmet Compliance System
            </div>
            <div class="user-info">
                <span class="role-badge">{role}</span>
                <span>{username}</span>
                <a href="/logout">Log out</a>
                <button class="theme-toggle" onclick="toggleTheme()"></button>
            </div>
        </div>
        <div class="dashboard-container">
            {body}
        </div>
        <div id="modal-overlay" class="modal-overlay">
            <div class="modal-box">
                <p id="modal-message"></p>
                <div class="modal-actions">
                    <button class="btn" onclick="modalCancel()">Cancel</button>
                    <button class="btn-primary" onclick="modalConfirm()">Confirm</button>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
