"""
Email Fetcher Module

Fetches emails from any IMAP-compatible email provider (Gmail, Yahoo, Outlook, etc.)
for spam analysis.
"""
import imaplib
import email
from email.header import decode_header
from email.utils import parsedate_to_datetime
from typing import List, Dict, Optional, Tuple
import re
from datetime import datetime


class EmailFetcher:
    """Fetches emails via IMAP protocol."""
    
    # Common IMAP servers for popular providers
    IMAP_SERVERS = {
        'gmail.com': 'imap.gmail.com',
        'googlemail.com': 'imap.gmail.com',
        'yahoo.com': 'imap.mail.yahoo.com',
        'yahoo.co.uk': 'imap.mail.yahoo.com',
        'outlook.com': 'imap-mail.outlook.com',
        'hotmail.com': 'imap-mail.outlook.com',
        'live.com': 'imap-mail.outlook.com',
        'icloud.com': 'imap.mail.me.com',
        'me.com': 'imap.mail.me.com',
        'protonmail.com': '127.0.0.1',  # Requires ProtonMail Bridge
        'aol.com': 'imap.aol.com',
    }
    
    def __init__(self):
        self.connection: Optional[imaplib.IMAP4_SSL] = None
        self.email_address: Optional[str] = None
        self.is_connected: bool = False
    
    def get_imap_server(self, email_address: str) -> str:
        """Auto-detect IMAP server from email domain."""
        domain = email_address.split('@')[-1].lower()
        
        if domain in self.IMAP_SERVERS:
            return self.IMAP_SERVERS[domain]
        
        # Default fallback - try imap.domain
        return f'imap.{domain}'
    
    def connect(self, email_address: str, password: str, 
                imap_server: Optional[str] = None, port: int = 993) -> Tuple[bool, str]:
        """
        Connect to IMAP server.
        
        Args:
            email_address: User's email address
            password: App password (not regular password for Gmail/Yahoo)
            imap_server: Optional IMAP server address (auto-detected if not provided)
            port: IMAP port (default 993 for SSL)
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Auto-detect server if not provided
            if not imap_server:
                imap_server = self.get_imap_server(email_address)
            
            # Connect with SSL
            self.connection = imaplib.IMAP4_SSL(imap_server, port)
            
            # Login
            self.connection.login(email_address, password)
            
            self.email_address = email_address
            self.is_connected = True
            
            return True, f"Connected to {imap_server}"
            
        except imaplib.IMAP4.error as e:
            error_msg = str(e)
            if 'AUTHENTICATIONFAILED' in error_msg or 'Invalid credentials' in error_msg.lower():
                return False, "Authentication failed. Please check your email and app password."
            return False, f"IMAP error: {error_msg}"
        except Exception as e:
            return False, f"Connection error: {str(e)}"
    
    def disconnect(self):
        """Disconnect from IMAP server."""
        if self.connection:
            try:
                self.connection.logout()
            except:
                pass
        self.connection = None
        self.email_address = None
        self.is_connected = False
    
    def _decode_header_value(self, header_value) -> str:
        """Decode email header value to string."""
        if not header_value:
            return ""
        
        decoded_parts = decode_header(header_value)
        result = []
        
        for part, charset in decoded_parts:
            if isinstance(part, bytes):
                try:
                    result.append(part.decode(charset or 'utf-8', errors='replace'))
                except:
                    result.append(part.decode('utf-8', errors='replace'))
            else:
                result.append(str(part))
        
        return ' '.join(result)
    
    def _get_email_body(self, msg) -> str:
        """Extract text body from email message."""
        body = ""
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition") or "")
                
                # Skip attachments
                if "attachment" in content_disposition:
                    continue
                
                # Get text content
                if content_type == "text/plain":
                    try:
                        payload = part.get_payload(decode=True)
                        charset = part.get_content_charset() or 'utf-8'
                        body = payload.decode(charset, errors='replace')
                        break  # Prefer plain text
                    except:
                        continue
                elif content_type == "text/html" and not body:
                    try:
                        payload = part.get_payload(decode=True)
                        charset = part.get_content_charset() or 'utf-8'
                        html_content = payload.decode(charset, errors='replace')
                        # Strip HTML tags for analysis
                        body = re.sub(r'<[^>]+>', ' ', html_content)
                        body = re.sub(r'\s+', ' ', body).strip()
                    except:
                        continue
        else:
            try:
                payload = msg.get_payload(decode=True)
                charset = msg.get_content_charset() or 'utf-8'
                
                if msg.get_content_type() == "text/html":
                    html_content = payload.decode(charset, errors='replace')
                    body = re.sub(r'<[^>]+>', ' ', html_content)
                    body = re.sub(r'\s+', ' ', body).strip()
                else:
                    body = payload.decode(charset, errors='replace')
            except:
                body = str(msg.get_payload())
        
        return body.strip()
    
    def fetch_recent(self, limit: int = 30, folder: str = 'INBOX') -> Tuple[List[Dict], str]:
        """
        Fetch the most recent emails from the specified folder.
        
        Args:
            limit: Maximum number of emails to fetch
            folder: Mailbox folder (default: INBOX)
            
        Returns:
            Tuple of (emails: List[Dict], error_message: str)
        """
        if not self.is_connected or not self.connection:
            return [], "Not connected to email server"
        
        try:
            # Select folder
            status, messages = self.connection.select(folder, readonly=True)
            if status != 'OK':
                return [], f"Could not select folder: {folder}"
            
            # Get total message count
            total_messages = int(messages[0])
            
            if total_messages == 0:
                return [], ""
            
            # Calculate range for most recent emails
            start = max(1, total_messages - limit + 1)
            end = total_messages
            
            # Fetch emails
            emails = []
            for i in range(end, start - 1, -1):  # Newest first
                try:
                    # Fetch email
                    status, msg_data = self.connection.fetch(str(i), '(RFC822)')
                    if status != 'OK':
                        continue
                    
                    # Parse email
                    raw_email = msg_data[0][1]
                    msg = email.message_from_bytes(raw_email)
                    
                    # Extract headers
                    subject = self._decode_header_value(msg.get('Subject', ''))
                    from_addr = self._decode_header_value(msg.get('From', ''))
                    date_str = msg.get('Date', '')
                    
                    # Parse date
                    try:
                        date = parsedate_to_datetime(date_str)
                        date_formatted = date.strftime('%Y-%m-%d %H:%M')
                    except:
                        date_formatted = date_str[:20] if date_str else 'Unknown'
                    
                    # Extract body
                    body = self._get_email_body(msg)
                    
                    # Combine subject and body for analysis
                    full_content = f"{subject}\n\n{body}" if subject else body
                    
                    emails.append({
                        'id': f"email_{i}",
                        'subject': subject or '(No Subject)',
                        'from': from_addr,
                        'date': date_formatted,
                        'content': full_content[:2000],  # Limit content length
                        'preview': (full_content[:200] + '...') if len(full_content) > 200 else full_content
                    })
                    
                except Exception as e:
                    print(f"Error fetching email {i}: {e}")
                    continue
            
            return emails, ""
            
        except Exception as e:
            return [], f"Error fetching emails: {str(e)}"
    
    def get_folders(self) -> List[str]:
        """Get list of available folders/mailboxes."""
        if not self.is_connected or not self.connection:
            return []
        
        try:
            status, folders = self.connection.list()
            if status != 'OK':
                return []
            
            folder_names = []
            for folder in folders:
                # Parse folder name from response
                match = re.search(r'"([^"]+)"$|(\S+)$', folder.decode())
                if match:
                    name = match.group(1) or match.group(2)
                    folder_names.append(name)
            
            return folder_names
        except:
            return []
