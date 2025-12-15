"""
Network Security Implementation - Anomaly Detection with Python
================================================================

This program analyzes network traffic and detects anomalies using both
rule-based and machine learning approaches.

Author: Network Security Coursework
Date: December 15, 2025
"""

import pandas as pd
import numpy as np
from scapy.all import rdpcap, IP, TCP, UDP, ICMP
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================

# Rule-based thresholds
MAX_PACKETS_PER_IP = 100  # Maximum packets from single IP in time window
TIME_WINDOW = 60  # Time window in seconds
PORT_SCAN_THRESHOLD = 20  # Number of different ports accessed = port scan
MAX_PACKET_SIZE = 1500  # Typical MTU size
SUSPICIOUS_PORTS = [23, 445, 3389, 4444, 5900]  # Telnet, SMB, RDP, Metasploit, VNC

# File paths
LOG_FILE = "alerts.log"
OUTPUT_CSV = "traffic_analysis.csv"

# ==================== HELPER FUNCTIONS ====================

def log_alert(message, alert_type="ANOMALY"):
    """
    Log anomaly alerts to file and console
    
    Args:
        message: Alert message to log
        alert_type: Type of alert (ANOMALY, PORT_SCAN, DOS, etc.)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{alert_type}] {message}"
    
    # Print to console
    print(f"\n{'='*80}")
    print(f"🚨 ALERT: {log_entry}")
    print(f"{'='*80}\n")
    
    # Write to log file
    with open(LOG_FILE, 'a') as f:
        f.write(log_entry + "\n")

# ==================== TRAFFIC CAPTURE & FEATURE EXTRACTION ====================

def extract_features_from_pcap(pcap_file):
    """
    Read PCAP file and extract relevant features for anomaly detection
    
    Args:
        pcap_file: Path to PCAP file
        
    Returns:
        DataFrame with extracted features
    """
    print(f"[*] Reading PCAP file: {pcap_file}")
    
    try:
        packets = rdpcap(pcap_file)
    except Exception as e:
        print(f"[!] Error reading PCAP file: {e}")
        return None
    
    print(f"[*] Total packets captured: {len(packets)}")
    
    # Initialize lists to store packet features
    traffic_data = []
    
    for i, packet in enumerate(packets):
        if IP in packet:
            # Extract timestamp
            timestamp = float(packet.time)
            
            # Extract IP layer information
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            protocol = packet[IP].proto
            packet_size = len(packet)
            ttl = packet[IP].ttl
            
            # Extract transport layer information
            src_port = 0
            dst_port = 0
            flags = ""
            
            if TCP in packet:
                src_port = packet[TCP].sport
                dst_port = packet[TCP].dport
                flags = str(packet[TCP].flags)
                protocol_name = "TCP"
            elif UDP in packet:
                src_port = packet[UDP].sport
                dst_port = packet[UDP].dport
                protocol_name = "UDP"
            elif ICMP in packet:
                protocol_name = "ICMP"
            else:
                protocol_name = "OTHER"
            
            # Store extracted features
            traffic_data.append({
                'timestamp': timestamp,
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'src_port': src_port,
                'dst_port': dst_port,
                'protocol': protocol,
                'protocol_name': protocol_name,
                'packet_size': packet_size,
                'ttl': ttl,
                'flags': flags
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(traffic_data)
    
    if not df.empty:
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        print(f"[✓] Successfully extracted features from {len(df)} packets")
    else:
        print("[!] No IP packets found in PCAP file")
    
    return df

# ==================== RULE-BASED ANOMALY DETECTION ====================

def detect_high_traffic_volume(df):
    """
    Detect anomalies based on high packet volume from single source
    (Potential DoS attack indicator)
    """
    print("\n[*] Running Rule 1: High Traffic Volume Detection...")
    
    # Group by source IP and count packets
    ip_counts = df.groupby('src_ip').size().reset_index(name='packet_count')
    
    # Find IPs exceeding threshold
    anomalous_ips = ip_counts[ip_counts['packet_count'] > MAX_PACKETS_PER_IP]
    
    if not anomalous_ips.empty:
        for _, row in anomalous_ips.iterrows():
            log_alert(
                f"High traffic volume detected from {row['src_ip']}: {row['packet_count']} packets",
                "DOS_ATTACK"
            )
    else:
        print("[✓] No high traffic volume anomalies detected")
    
    return anomalous_ips

def detect_port_scanning(df):
    """
    Detect port scanning behavior
    (Single IP accessing many different ports)
    """
    print("\n[*] Running Rule 2: Port Scan Detection...")
    
    # Group by source IP and count unique destination ports
    port_scan_data = df.groupby('src_ip')['dst_port'].nunique().reset_index(name='unique_ports')
    
    # Find IPs accessing many ports
    port_scanners = port_scan_data[port_scan_data['unique_ports'] > PORT_SCAN_THRESHOLD]
    
    if not port_scanners.empty:
        for _, row in port_scanners.iterrows():
            log_alert(
                f"Port scanning detected from {row['src_ip']}: {row['unique_ports']} different ports accessed",
                "PORT_SCAN"
            )
    else:
        print("[✓] No port scanning anomalies detected")
    
    return port_scanners

def detect_suspicious_ports(df):
    """
    Detect access to suspicious/vulnerable ports
    """
    print("\n[*] Running Rule 3: Suspicious Port Detection...")
    
    # Filter traffic to suspicious ports
    suspicious_traffic = df[df['dst_port'].isin(SUSPICIOUS_PORTS)]
    
    if not suspicious_traffic.empty:
        for port in SUSPICIOUS_PORTS:
            port_traffic = suspicious_traffic[suspicious_traffic['dst_port'] == port]
            if not port_traffic.empty:
                unique_sources = port_traffic['src_ip'].nunique()
                log_alert(
                    f"Suspicious port {port} accessed by {unique_sources} unique IP(s): {len(port_traffic)} connections",
                    "SUSPICIOUS_PORT"
                )
    else:
        print("[✓] No suspicious port access detected")
    
    return suspicious_traffic

def detect_unusual_packet_sizes(df):
    """
    Detect packets with unusual sizes (potential data exfiltration or attacks)
    """
    print("\n[*] Running Rule 4: Unusual Packet Size Detection...")
    
    # Calculate statistics
    mean_size = df['packet_size'].mean()
    std_size = df['packet_size'].std()
    
    # Define unusual as packets > 3 standard deviations from mean
    threshold_high = mean_size + (3 * std_size)
    threshold_low = mean_size - (3 * std_size)
    
    unusual_packets = df[(df['packet_size'] > threshold_high) | (df['packet_size'] < threshold_low)]
    
    if not unusual_packets.empty:
        log_alert(
            f"Detected {len(unusual_packets)} packets with unusual sizes (Mean: {mean_size:.2f}, Std: {std_size:.2f})",
            "UNUSUAL_SIZE"
        )
    else:
        print("[✓] No unusual packet sizes detected")
    
    return unusual_packets

# ==================== MACHINE LEARNING ANOMALY DETECTION ====================

def ml_anomaly_detection(df):
    """
    Use Isolation Forest algorithm to detect anomalies in network traffic
    
    Isolation Forest is an unsupervised learning algorithm that isolates
    anomalies by randomly selecting features and split values.
    """
    print("\n[*] Running Machine Learning Anomaly Detection (Isolation Forest)...")
    
    # Select numerical features for ML model
    features_for_ml = ['packet_size', 'ttl', 'src_port', 'dst_port', 'protocol']
    
    # Prepare data
    df_ml = df[features_for_ml].copy()
    
    # Handle any missing values
    df_ml = df_ml.fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_ml)
    
    # Train Isolation Forest
    # contamination: expected proportion of outliers (10%)
    iso_forest = IsolationForest(
        contamination=0.1,
        random_state=42,
        n_estimators=100
    )
    
    # Predict anomalies (-1 for anomalies, 1 for normal)
    predictions = iso_forest.fit_predict(X_scaled)
    
    # Add predictions to dataframe
    df['ml_anomaly'] = predictions
    df['anomaly_score'] = iso_forest.score_samples(X_scaled)
    
    # Get anomalous packets
    ml_anomalies = df[df['ml_anomaly'] == -1]
    
    print(f"[✓] ML Detection Complete: {len(ml_anomalies)} anomalies detected ({len(ml_anomalies)/len(df)*100:.2f}%)")
    
    if not ml_anomalies.empty:
        # Log top anomalies
        top_anomalies = ml_anomalies.nlargest(5, 'anomaly_score', keep='first')
        log_alert(
            f"ML model detected {len(ml_anomalies)} anomalous packets",
            "ML_DETECTION"
        )
        
        for idx, row in top_anomalies.iterrows():
            log_alert(
                f"  - {row['src_ip']}:{row['src_port']} -> {row['dst_ip']}:{row['dst_port']} "
                f"[{row['protocol_name']}] Size: {row['packet_size']} bytes",
                "ML_ANOMALY"
            )
    
    return ml_anomalies

# ==================== VISUALIZATION ====================

def visualize_traffic(df):
    """
    Create visualizations of traffic patterns and anomalies
    """
    print("\n[*] Generating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Network Traffic Analysis & Anomaly Detection', fontsize=16, fontweight='bold')
    
    # 1. Protocol Distribution
    protocol_counts = df['protocol_name'].value_counts()
    axes[0, 0].pie(protocol_counts.values, labels=protocol_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Protocol Distribution')
    
    # 2. Top Source IPs
    top_sources = df['src_ip'].value_counts().head(10)
    axes[0, 1].barh(range(len(top_sources)), top_sources.values)
    axes[0, 1].set_yticks(range(len(top_sources)))
    axes[0, 1].set_yticklabels(top_sources.index)
    axes[0, 1].set_xlabel('Packet Count')
    axes[0, 1].set_title('Top 10 Source IPs by Traffic Volume')
    axes[0, 1].invert_yaxis()
    
    # 3. Packet Size Distribution
    axes[1, 0].hist(df['packet_size'], bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Packet Size (bytes)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Packet Size Distribution')
    axes[1, 0].axvline(df['packet_size'].mean(), color='red', linestyle='--', label=f'Mean: {df["packet_size"].mean():.0f}')
    axes[1, 0].legend()
    
    # 4. Port Distribution
    port_data = df[df['dst_port'] > 0]['dst_port'].value_counts().head(15)
    axes[1, 1].bar(range(len(port_data)), port_data.values, color='steelblue')
    axes[1, 1].set_xticks(range(len(port_data)))
    axes[1, 1].set_xticklabels(port_data.index, rotation=45)
    axes[1, 1].set_xlabel('Port Number')
    axes[1, 1].set_ylabel('Packet Count')
    axes[1, 1].set_title('Top 15 Destination Ports')
    
    plt.tight_layout()
    plt.savefig('traffic_analysis.png', dpi=300, bbox_inches='tight')
    print("[✓] Visualization saved as 'traffic_analysis.png'")
    
    # Create anomaly-specific visualization if ML was run
    if 'ml_anomaly' in df.columns:
        fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5))
        fig2.suptitle('Machine Learning Anomaly Detection Results', fontsize=16, fontweight='bold')
        
        # Anomaly score distribution
        axes2[0].hist(df['anomaly_score'], bins=50, edgecolor='black', alpha=0.7)
        axes2[0].set_xlabel('Anomaly Score')
        axes2[0].set_ylabel('Frequency')
        axes2[0].set_title('Anomaly Score Distribution')
        axes2[0].axvline(df[df['ml_anomaly'] == -1]['anomaly_score'].max(), 
                        color='red', linestyle='--', label='Anomaly Threshold')
        axes2[0].legend()
        
        # Normal vs Anomaly comparison
        anomaly_counts = df['ml_anomaly'].value_counts()
        labels = ['Normal', 'Anomaly']
        colors = ['green', 'red']
        axes2[1].bar(labels, [anomaly_counts.get(1, 0), anomaly_counts.get(-1, 0)], color=colors, alpha=0.7)
        axes2[1].set_ylabel('Packet Count')
        axes2[1].set_title('Normal vs Anomalous Packets')
        
        for i, (label, count) in enumerate(zip(labels, [anomaly_counts.get(1, 0), anomaly_counts.get(-1, 0)])):
            axes2[1].text(i, count, str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('ml_anomaly_analysis.png', dpi=300, bbox_inches='tight')
        print("[✓] ML visualization saved as 'ml_anomaly_analysis.png'")

# ==================== MAIN ANALYSIS FUNCTION ====================

def analyze_traffic(pcap_file):
    """
    Main function to orchestrate traffic analysis and anomaly detection
    
    Args:
        pcap_file: Path to PCAP file to analyze
    """
    print("\n" + "="*80)
    print(" NETWORK TRAFFIC ANOMALY DETECTION SYSTEM")
    print("="*80 + "\n")
    
    # Clear previous log file
    open(LOG_FILE, 'w').close()
    
    # Step 1: Extract features from PCAP
    df = extract_features_from_pcap(pcap_file)
    
    if df is None or df.empty:
        print("[!] No data to analyze. Exiting.")
        return
    
    # Save extracted features to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[✓] Traffic data saved to '{OUTPUT_CSV}'")
    
    # Display basic statistics
    print("\n" + "-"*80)
    print("TRAFFIC STATISTICS")
    print("-"*80)
    print(f"Total Packets: {len(df)}")
    print(f"Unique Source IPs: {df['src_ip'].nunique()}")
    print(f"Unique Destination IPs: {df['dst_ip'].nunique()}")
    print(f"Protocol Distribution:\n{df['protocol_name'].value_counts()}")
    print(f"Average Packet Size: {df['packet_size'].mean():.2f} bytes")
    print("-"*80 + "\n")
    
    # Step 2: Rule-Based Anomaly Detection
    print("\n" + "="*80)
    print(" RULE-BASED ANOMALY DETECTION")
    print("="*80)
    
    high_traffic = detect_high_traffic_volume(df)
    port_scans = detect_port_scanning(df)
    suspicious_ports = detect_suspicious_ports(df)
    unusual_sizes = detect_unusual_packet_sizes(df)
    
    # Step 3: Machine Learning Anomaly Detection
    print("\n" + "="*80)
    print(" MACHINE LEARNING ANOMALY DETECTION")
    print("="*80)
    
    ml_anomalies = ml_anomaly_detection(df)
    
    # Step 4: Generate Visualizations
    visualize_traffic(df)
    
    # Step 5: Summary Report
    print("\n" + "="*80)
    print(" DETECTION SUMMARY")
    print("="*80)
    print(f"High Traffic Volume Anomalies: {len(high_traffic)}")
    print(f"Port Scanning Attempts: {len(port_scans)}")
    print(f"Suspicious Port Access: {len(suspicious_ports)}")
    print(f"Unusual Packet Sizes: {len(unusual_sizes)}")
    print(f"ML-Detected Anomalies: {len(ml_anomalies)}")
    print(f"\nTotal Anomalies Detected: {len(high_traffic) + len(port_scans) + len(suspicious_ports) + len(unusual_sizes) + len(ml_anomalies)}")
    print(f"All alerts logged to: {LOG_FILE}")
    print("="*80 + "\n")

# ==================== SAMPLE DATA GENERATOR ====================

def generate_sample_pcap():
    """
    Generate sample network traffic for testing
    (Use this if you don't have a PCAP file)
    """
    from scapy.all import wrpcap, Ether, IP, TCP, UDP
    
    print("[*] Generating sample PCAP file...")
    
    packets = []
    
    # Normal traffic
    for i in range(200):
        pkt = Ether()/IP(src=f"192.168.1.{np.random.randint(1, 20)}", 
                        dst=f"192.168.1.{np.random.randint(20, 50)}")/TCP(
                            sport=np.random.randint(1024, 65535),
                            dport=np.random.choice([80, 443, 22, 53])
                        )
        packets.append(pkt)
    
    # Port scan simulation
    scanner_ip = "10.0.0.100"
    for port in range(1, 100):
        pkt = Ether()/IP(src=scanner_ip, dst="192.168.1.50")/TCP(sport=54321, dport=port, flags="S")
        packets.append(pkt)
    
    # DoS simulation
    for i in range(150):
        pkt = Ether()/IP(src="203.0.113.50", dst="192.168.1.10")/TCP(sport=12345, dport=80, flags="S")
        packets.append(pkt)
    
    # Suspicious port access
    for i in range(20):
        pkt = Ether()/IP(src=f"172.16.0.{np.random.randint(1, 10)}", 
                        dst="192.168.1.100")/TCP(sport=np.random.randint(1024, 65535), dport=445)
        packets.append(pkt)
    
    wrpcap("sample_traffic.pcap", packets)
    print("[✓] Sample PCAP file 'sample_traffic.pcap' created successfully")
    return "sample_traffic.pcap"

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    import sys
    
    print("\n🔒 Network Security - Anomaly Detection System 🔒\n")
    
    # Check if PCAP file is provided
    if len(sys.argv) > 1:
        pcap_file = sys.argv[1]
    else:
        print("[*] No PCAP file provided.")
        choice = input("Would you like to:\n  1. Generate sample traffic\n  2. Provide PCAP file path\nChoice (1/2): ")
        
        if choice == "1":
            pcap_file = generate_sample_pcap()
        else:
            pcap_file = input("Enter path to PCAP file: ")
    
    # Run analysis
    try:
        analyze_traffic(pcap_file)
        print("\n[✓] Analysis complete! Check the generated files:")
        print(f"  - {LOG_FILE} (Alert logs)")
        print(f"  - {OUTPUT_CSV} (Extracted traffic data)")
        print("  - traffic_analysis.png (Traffic visualizations)")
        print("  - ml_anomaly_analysis.png (ML results)")
    except Exception as e:
        print(f"\n[!] Error during analysis: {e}")
        import traceback
        traceback.print_exc()
