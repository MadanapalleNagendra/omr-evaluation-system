"""
OMR System Authentication Layer
Handles user login and session management
"""

import streamlit as st
from datetime import datetime
import json
import os

USERS_FILE = "users.json"

def initialize_auth_state():
    """Initialize authentication session state"""
    defaults = {
        'logged_in': False,
        'username': None,
        'login_time': None,
        'failed_attempts': 0,
        'last_attempt': None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def show_login_interface():
    """Display login interface"""
    st.markdown("""
    <div style='text-align: center; padding: 3rem 1rem;'>
        <div style='font-size: 4rem; color: #1f77b4; margin-bottom: 1rem;'>🔒</div>
        <h1 style='color: #1f77b4; margin-bottom: 0.5rem;'>OMR Evaluation System</h1>
        <h3 style='color: #666; margin-bottom: 2rem;'>Secure Access Required</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Login container
    login_container = st.container()
    with login_container:
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### 🔐 Authentication")
            
            # Rate limiting check
            if st.session_state.get('failed_attempts', 0) >= 3:
                now = datetime.now()
                last_attempt = st.session_state.get('last_attempt', now)
                time_diff = (now - last_attempt).total_seconds()
                
                if time_diff < 300:  # 5 minutes lockout
                    remaining = int(300 - time_diff)
                    st.error(f"⏳ **Account temporarily locked**")
                    st.info(f"Too many failed attempts. Please wait {remaining} seconds before trying again.")
                    if st.button("🔄 Reset Counter", type='secondary'):
                        st.session_state['failed_attempts'] = 0
                        st.session_state['last_attempt'] = None
                        st.rerun()
                    return False
                else:
                    st.session_state['failed_attempts'] = 0
                    st.rerun()
            
            username = st.text_input('👤 Username', placeholder="Enter your username")
            password = st.text_input('🔑 Password', type='password', placeholder="Enter your password")
            
            col_btn1, _, col_btn2 = st.columns([2, 1, 2])
            with col_btn1:
                if st.button('🚀 Sign In', type='primary', use_container_width=True, disabled=not username or not password):
                    if username == 'admin' and password == 'admin':
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = username
                        st.session_state['login_time'] = datetime.now()
                        st.session_state['failed_attempts'] = 0
                        st.success(f'✅ Welcome back, {username}!')
                        st.balloons()
                        st.rerun()
                    else:
                        st.session_state['failed_attempts'] = st.session_state.get('failed_attempts', 0) + 1
                        st.session_state['last_attempt'] = datetime.now()
                        st.error('❌ Invalid credentials. Please try again.')
            
            # Help section
            with st.expander("ℹ️ Demo Access", expanded=False):
                st.info("""
                **🔓 Demo Credentials:**
                - **Username:** `admin`
                - **Password:** `admin`
                
                **For Production Use:**
                - Update credentials in `login.py`
                - Implement database authentication
                - Add role-based access control
                - Enable multi-factor authentication
                """)
    
    return st.session_state.get('logged_in', False)

def show_user_dashboard():
    """Display user dashboard with logout functionality"""
    # User info in sidebar
    with st.sidebar:
        st.markdown("---")
        if st.session_state.get('logged_in', False):
            st.markdown(f"""
            <div style='text-align: center; padding: 1rem; background: #e3f2fd; border-radius: 0.5rem;'>
                <h3 style='margin: 0; color: #1976d2;'>👋 {st.session_state.get('username', 'User')}</h3>
                <p style='margin: 0.5rem 0 0 0; color: #666; font-size: 0.9rem;'>
                    Active since {st.session_state.get('login_time', datetime.now()).strftime('%H:%M:%S')}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button('🚪 Logout', type='secondary', use_container_width=True):
                # Clear session
                for key in list(st.session_state.keys()):
                    if key not in ['_sentry_event_id']:
                        del st.session_state[key]
                st.session_state['logged_in'] = False
                st.rerun()
        else:
            st.warning("⚠️ Not authenticated")
    
    return True

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {"admin": "admin"}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def user_management(users):
    st.sidebar.markdown("---")
    st.sidebar.subheader("👤 User Management (Admin Only)")
    # Add user
    new_user = st.sidebar.text_input("New Username", key="add_user")
    new_pass = st.sidebar.text_input("New Password", type="password", key="add_pass")
    if st.sidebar.button("Add User"):
        if new_user and new_pass:
            if new_user in users:
                st.sidebar.warning("User already exists!")
            else:
                users[new_user] = new_pass
                save_users(users)
                st.sidebar.success(f"User '{new_user}' added.")
        else:
            st.sidebar.warning("Enter username and password.")
    # Remove user
    del_user = st.sidebar.selectbox("Delete User", [u for u in users if u != "admin"], key="del_user")
    if st.sidebar.button("Delete User"):
        if del_user in users:
            del users[del_user]
            save_users(users)
            st.sidebar.success(f"User '{del_user}' deleted.")
    # Change password (for any user)
    st.sidebar.subheader("Change Your Password")
    if st.session_state.get("username"):
        old_pass = st.sidebar.text_input("Old Password", type="password", key="oldpw")
        new_pass2 = st.sidebar.text_input("New Password", type="password", key="newpw")
        if st.sidebar.button("Change Password"):
            user = st.session_state["username"]
            if users[user] == old_pass and new_pass2:
                users[user] = new_pass2
                save_users(users)
                st.sidebar.success("Password changed.")
            else:
                st.sidebar.error("Incorrect old password or empty new password.")

# Main execution
def main():
    users = load_users()
    initialize_auth_state()
    
    # Check authentication
    if not show_user_dashboard():
        # Show login page
        if not show_login_interface():
            return
    else:
        # Show user management for admin
        if st.session_state.get("username") == "admin":
            user_management(users)
        # Show main OMR application
        show_omr_application()

def show_omr_application():
    """Display the main OMR application"""
    # Custom CSS for main app
    # Import main app UI functions from app.py for shared use
    from app import (
        load_processor,
        sidebar_configuration,
        file_upload_section,
        process_images,
        display_help_section,
        display_results_summary,
        display_scoring_results,
        display_individual_results,
        download_results
    )

    # Custom CSS for main app
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            color: #1f77b4;
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        .success-message { color: #28a745; font-weight: bold; }
        .error-message { color: #dc3545; font-weight: bold; }
        .warning-message { color: #ffc107; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

    # Welcome message
    st.markdown('<h1 class="main-header">\ud83d\udccb OMR Evaluation Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Load processor
    if not load_processor():
        st.error("Failed to load OMR processing engine")
        return

    # Sidebar configuration
    config_options = sidebar_configuration()

    # Main content
    col_main, col_sidebar = st.columns([3, 1])

    with col_main:
        uploaded_files = file_upload_section()

        if uploaded_files:
            if st.button("\ud83d\ude80 Process OMR Sheets", type="primary", use_container_width=True):
                st.session_state.batch_results = None
                st.rerun()

                with st.spinner(f"Processing {len(uploaded_files)} sheets..."):
                    results = process_images(uploaded_files, config_options)

                    if results and 'error' not in results:
                        st.session_state.batch_results = results
                        st.rerun()

    with col_sidebar:
        display_help_section()

    # Results display
    if st.session_state.get('batch_results'):
        display_results_summary(st.session_state.batch_results)
        display_scoring_results(st.session_state.batch_results)
        display_individual_results(st.session_state.batch_results)
        download_results(st.session_state.batch_results, config_options)
# Include all your existing functions here (load_processor, sidebar_configuration, etc.)
# ... [Include all the functions from your original app.py] ...

if __name__ == "__main__":
    main()