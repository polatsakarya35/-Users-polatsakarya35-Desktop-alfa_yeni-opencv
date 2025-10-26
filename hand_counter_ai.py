#!/usr/bin/env python3
"""
MediaPipe Simple System - Mac M4
================================
Class olmadan basit ve temiz kod
"""

import cv2 # OpenCV
import numpy as np # NumPy (gerekli değil ama işlevsellik için kullanılır)
import mediapipe as mp # MediaPipe modülü = Vücut ve el tespiti için
import time # FPS hesaplaması için

# Global değişkenler
current_fps = 0
fps_start_time = time.time()
fps_frame_count = 0 
current_number = 0

# MediaPipe modelleri
mp_pose = mp.solutions.pose # Vücut tespiti için
mp_hands = mp.solutions.hands # El tespiti için
mp_drawing = mp.solutions.drawing_utils # Drawing utils modülü = Çizim işlemleri için
mp_drawing_styles = mp.solutions.drawing_styles # Drawing styles modülü = Çizim stilleri için

# Pose modeli
pose = mp_pose.Pose( # Pose modeli = Vücut tespiti için
    static_image_mode=False, # Statik görüntü modu = False = Video akışı için
    model_complexity=1, # Model karmaşıklığı = 1 = Orta karmaşıklık
    enable_segmentation=False, # Segmentasyon etkinleştirme = False = Segmentasyon etkinleştirme
    min_detection_confidence=0.7, # Minimum tespit güveni = 0.7 = 70% orta seviye hata
    min_tracking_confidence=0.5 # Minimum takip güveni = 0.5 = 50% orta seviye hata
)

# El tespiti modeli
hands = mp_hands.Hands( # Hands modeli = El tespiti için
    static_image_mode=False, # Statik görüntü modu = False = Video akışı için
    max_num_hands=2, # Maksimum el sayısı = 2 = İki el tespiti
    min_detection_confidence=0.7, # Minimum tespit güveni = 0.7 = 70%
    min_tracking_confidence=0.5 # Minimum takip güveni = 0.5 = 50%
)

# Sayı sözlüğü
number_gestures = { # Sayı sözlüğü = Sayıların adları için
    0: '0️⃣ Sıfır', 1: '1️⃣ Bir', 2: '2️⃣ İki', 3: '3️⃣ Üç', 4: '4️⃣ Dört', 5: '5️⃣ Beş',
    6: '6️⃣ Altı', 7: '7️⃣ Yedi', 8: '8️⃣ Sekiz', 9: '9️⃣ Dokuz', 10: '🔟 On'
}

def detect_pose(frame): # Vücut pose tespiti için
    """Vücut pose tespiti."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # RGB formatına çevir mediapipe için
    results = pose.process(rgb_frame) # Pose modeli ile işle "process" = görüntüyü analiz eder, 33 vucut noktasını bulur
    return results # Sonuçları döndür

def detect_hands(frame):
    """El tespiti - 21 landmark."""
    hands_data = [] 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks: # Eğer el tespiti yapıldıysa
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):  # multi_hand_landmarks = el tespiti noktaları
            hand_label = results.multi_handedness[idx].classification[0].label  # multi_handedness = el tespiti yönü (mp nin özelliği) "classification" = el tespiti yönünün adı bu dosyaya girip "label" = el tespiti yönünün adı nı alıyor.
            # hand_label = el tespiti yönünün adı (sol el veya sağ el)
            landmarks = [] # landmarks = el tespiti noktaları
            for landmark in hand_landmarks.landmark: # landmark = el tespiti noktası
                h, w, c = frame.shape
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                z = landmark.z
                landmarks.append((x, y, z))
            
            hands_data.append({
                'hand': hand_label,
                'landmarks': landmarks,
                'confidence': 0.95
            })
    
    return hands_data

def recognize_number(landmarks):
    """Tek el sayı tanıma - 1-5."""
    if len(landmarks) < 21:
        return 0
    
    # Parmak uçları
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    # Parmak eklemleri
    thumb_mcp = landmarks[2]
    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]
    ring_mcp = landmarks[13]
    pinky_mcp = landmarks[17]
    
    # Parmak sayısı
    fingers_count = 0
    
    # Başparmak
    if thumb_tip[0] > thumb_mcp[0] + 10: # thumb_tip[0] = başparmak uçu x, thumb_mcp[0] = başparmak ekleme x
        fingers_count += 1 # fingers_count = parmak sayısı
    
    # Diğer parmaklar
    if index_tip[1] < index_mcp[1] - 15: # index_tip[1] = index parmak uçu y, index_mcp[1] = index parmak ekleme y
        fingers_count += 1
    if middle_tip[1] < middle_mcp[1] - 15: # middle_tip[1] = middle parmak uçu y, middle_mcp[1] = middle parmak ekleme y
        fingers_count += 1
    if ring_tip[1] < ring_mcp[1] - 15: # ring_tip[1] = ring parmak uçu y, ring_mcp[1] = ring parmak ekleme y
        fingers_count += 1
    if pinky_tip[1] < pinky_mcp[1] - 15: # pinky_tip[1] = pinky parmak uçu y, pinky_mcp[1] = pinky parmak ekleme y
        fingers_count += 1
    
    return fingers_count if fingers_count <= 5 else 0

def recognize_dual_hand_number(hands_data):
    """İki el sayı tanıma - 6-10."""
    if len(hands_data) < 2:
        return 0
    
    total_fingers = 0
    
    for hand in hands_data:
        landmarks = hand['landmarks'] # landmarks = el tespiti noktaları
        hand_label = hand['hand'] # hand_label = el tespiti yönünün adı (sol el veya sağ el)
        
        if len(landmarks) >= 21:
            # Parmak uçları
            thumb_tip = landmarks[4] # thumb_tip = başparmak uçu
            index_tip = landmarks[8] # index_tip = index parmak uçu
            middle_tip = landmarks[12] # middle_tip = middle parmak uçu
            ring_tip = landmarks[16] # ring_tip = ring parmak uçu
            pinky_tip = landmarks[20] # pinky_tip = pinky parmak uçu
            
            # Parmak eklemleri
            thumb_mcp = landmarks[2] # thumb_mcp = başparmak ekleme
            index_mcp = landmarks[5] # index_mcp = index parmak ekleme
            middle_mcp = landmarks[9] # middle_mcp = middle parmak ekleme           
            ring_mcp = landmarks[13] # ring_mcp = ring parmak ekleme
            pinky_mcp = landmarks[17] # pinky_mcp = pinky parmak ekleme
            
            # Bu elin parmak sayısı
            hand_fingers = 0 # hand_fingers = elin parmak sayısı
            
            # Başparmak kontrolü
            if hand_label == 'Left':
                # Sol el için hassas kontrol
                if abs(thumb_tip[0] - thumb_mcp[0]) > 30: # thumb_tip[0] = başparmak uçu x, thumb_mcp[0] = başparmak ekleme x
                    hand_fingers += 1
            else:
                # Sağ el için normal kontrol
                if thumb_tip[0] > thumb_mcp[0]:
                    hand_fingers += 1
            
            # Diğer parmaklar
            if index_tip[1] < index_mcp[1]:
                hand_fingers += 1
            if middle_tip[1] < middle_mcp[1]:
                hand_fingers += 1
            if ring_tip[1] < ring_mcp[1]:
                hand_fingers += 1
            if pinky_tip[1] < pinky_mcp[1]:
                hand_fingers += 1
            
            total_fingers += hand_fingers
    
    # 6-10 arası sayılar
    if total_fingers == 6:
        return 6
    elif total_fingers == 7:
        return 7
    elif total_fingers == 8:
        return 8
    elif total_fingers == 9:
        return 9
    elif total_fingers == 10:
        return 10
    
    return 0

def draw_pose(frame, pose_results): 
    """Vücut pose çizimi."""
    if pose_results.pose_landmarks: # Eğer vücut pose tespiti yapıldıysa "pose_landmarks" = vücut pose tespiti noktaları
        mp_drawing.draw_landmarks( # Vücut pose tespiti noktalarını çiz
            frame, # frame = görüntü
            pose_results.pose_landmarks, # pose_results.pose_landmarks = vücut pose tespiti noktaları
            mp_pose.POSE_CONNECTIONS, # mp_pose.POSE_CONNECTIONS = vücut pose tespiti bağlantıları
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style() # mp_drawing_styles.get_default_pose_landmarks_style() = vücut pose tespiti noktalarının çizimi için
        ) # landm

def draw_hands(frame, hands_data):
    """El çizimi - 21 landmark."""
    for hand in hands_data:
        landmarks = hand['landmarks'] # hand['landmarks'] = el tespiti noktaları
        hand_label = hand['hand'] # hand['hand'] = el tespiti yönünün adı (sol el veya sağ el)
        
        # El rengi
        color = (0, 255, 0) if hand_label == 'Left' else (255, 0, 0)
        outline_color = (0, 180, 0) if hand_label == 'Left' else (180, 0, 0)
        
        # 21 landmark noktasını çiz
        for idx, (x, y, z) in enumerate(landmarks):
            # Parmak uçları (4, 8, 12, 16, 20)
            if idx in [4, 8, 12, 16, 20]: # cv2.circle = daire çiz
                cv2.circle(frame, (x, y), 8, outline_color, 3) # outline_color = el rengi
                cv2.circle(frame, (x, y), 6, color, -1) # color = el rengi
                cv2.circle(frame, (x, y), 3, (255, 255, 255), -1) # (255, 255, 255) = beyaz renk
            else:
                cv2.circle(frame, (x, y), 4, outline_color, 2)
                cv2.circle(frame, (x, y), 3, color, -1)
        
        # El bağlantıları
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
        ]
        
        for start_idx, end_idx in connections: # connections = el bağlantıları
            if start_idx < len(landmarks) and end_idx < len(landmarks): # start_idx = başlangıç noktası, end_idx = bitiş noktası, landmarks = el tespiti noktaları
                start_point = (landmarks[start_idx][0], landmarks[start_idx][1]) # start_point = başlangıç noktası
                end_point = (landmarks[end_idx][0], landmarks[end_idx][1]) # end_point = bitiş noktası
                cv2.line(frame, start_point, end_point, color, 2) # cv2.line = çizgi çiz

def update_fps():
    """FPS hesapla."""
    global current_fps, fps_start_time, fps_frame_count
    
    fps_frame_count += 1
    current_time = time.time()
    
    if current_time - fps_start_time >= 1.0:
        current_fps = fps_frame_count
        fps_frame_count = 0
        fps_start_time = current_time

def draw_number(frame, number):
    """Sayıyı süper havalı tasarımla göster."""
    if number < 0:
        return
    
    number_text = f"{number}"
    
    # Sayı için büyük boyut
    number_size = 6
    number_thickness = 8
    
    # Sağ üst köşe pozisyonu
    text_size = cv2.getTextSize(
        number_text, # number_text = sayı
        cv2.FONT_HERSHEY_SIMPLEX, # cv2.FONT_HERSHEY_SIMPLEX = font tipi
        number_size, # number_size = sayı boyutu
        number_thickness)[0] # number_thickness = sayı kalınlığı
    
    text_x = frame.shape[1] - text_size[0] - 40 # text_x = sayı x konumu
    text_y = 100
    
    # Büyük arka plan kutusu
    padding = 40
    box_x1 = text_x - padding
    box_y1 = text_y - text_size[1] - padding
    box_x2 = text_x + text_size[0] + padding
    box_y2 = text_y + padding
    
    # Gradient efekti - Çok katmanlı
    for i in range(30): # "Sayı etrafında 30 katmanlı gradient kenarlık oluştur"

        alpha = 1.0 - (i / 30.0)
        color_intensity = int(255 * alpha)
        # Renk geçişi: Yeşil -> Mavi -> Mor
        if i < 10:
            color = (0, color_intensity, 0)  # Yeşil
        elif i < 20:
            color = (0, 0, color_intensity)  # Mavi
        else:
            color = (color_intensity, 0, color_intensity)  # Mor
        
        cv2.rectangle(frame, # cv2.rectangle = dikdörtgen çiz
                    (box_x1 + i, box_y1 + i), # box_x1 + i = başlangıç x, box_y1 + i = başlangıç y
                    (box_x2 - i, box_y2 - i), # box_x2 - i = bitiş x, box_y2 - i = bitiş y
                    color, 2) # color = renk, 2 = kalınlık
    
    # Ana arka plan - Koyu siyah
    cv2.rectangle(frame, 
    (box_x1, box_y1), 
    (box_x2, box_y2), 
    (0, 0, 0), # (0, 0, 0) = siyah renk,
    -1 # -1 = tamamı doldur
    )
    
    # Neon kenarlık efekti - Çok katmanlı
    for i in range(5):
        neon_color = (0, 255, 255) if i % 2 == 0 else (255, 0, 255)
        cv2.rectangle(frame, 
                    (box_x1 - i, box_y1 - i),
                    (box_x2 + i, box_y2 + i),
                    neon_color, 2)
    
    # Sayıyı yazdır - Parlak beyaz
    cv2.putText(frame, number_text, (text_x, text_y),  # cv2.putText = yazı yazdır  number_thickness = sayı kalınlığı
               cv2.FONT_HERSHEY_SIMPLEX, number_size, (255, 255, 255), number_thickness)
    
    # Gölge efekti
    cv2.putText(frame, number_text, (text_x + 3, text_y + 3), 
               cv2.FONT_HERSHEY_SIMPLEX, number_size, (0, 0, 0), number_thickness)
    
    # Alt kısımda küçük açıklama
    desc_text = f"Parmak: {number}"
    desc_size = 1.2
    desc_thickness = 2
    desc_size_calc = cv2.getTextSize(desc_text, cv2.FONT_HERSHEY_SIMPLEX, desc_size, desc_thickness)[0]
    desc_x = text_x + (text_size[0] - desc_size_calc[0]) // 2
    desc_y = box_y2 + 30
    
    # Açıklama arka planı
    desc_padding = 10
    cv2.rectangle(frame, #rectangle = dikdörtgen çiz
                (desc_x - desc_padding, desc_y - desc_size_calc[1] - desc_padding),
                (desc_x + desc_size_calc[0] + desc_padding, desc_y + desc_padding),
                (20, 20, 20), -1)
    
    # Açıklama yazısı
    cv2.putText(frame, desc_text, (desc_x, desc_y), # cv2.putText = yazı yazdır  desc_thickness = açıklama kalınlığı
               cv2.FONT_HERSHEY_SIMPLEX, desc_size, (0, 255, 255), desc_thickness)

def main():
    """Ana fonksiyon."""
    global current_number
    
    print("🤖 MediaPipe Simple System - Mac M4")
    print("===================================")
    print("✅ MediaPipe basit sistemi hazır!")
    print("🎬 ESC tuşu ile çıkış")
    print("\n📖 KULLANIM KILAVUZU:")
    print("===================")
    print("0️⃣ Sıfır: Hiç el kaldırmayın veya tüm parmakları kapatın")
    print("1️⃣ Bir: 1 parmak kaldırın")
    print("2️⃣ İki: 2 parmak kaldırın")
    print("3️⃣ Üç: 3 parmak kaldırın")
    print("4️⃣ Dört: 4 parmak kaldırın")
    print("5️⃣ Beş: 5 parmak kaldırın")
    print("6️⃣ Altı: Sol el 5 + Sağ el 1 parmak")
    print("7️⃣ Yedi: Sol el 5 + Sağ el 2 parmak")
    print("8️⃣ Sekiz: Sol el 5 + Sağ el 3 parmak")
    print("9️⃣ Dokuz: Sol el 5 + Sağ el 4 parmak")
    print("🔟 On: Sol el 5 + Sağ el 5 parmak")
    print("\n💡 İPUCU: Ellerinizi kameraya net gösterin!")
    
    cap = cv2.VideoCapture(0) # cv2.VideoCapture = kamera aç
    
    if not cap.isOpened(): # Eğer kamera açılamadıysa
        print("❌ Kamera açılamadı!")
        return
    
    print("✅ Kamera başarıyla açıldı!")
    print("🎬 MediaPipe basit sistemi başlatılıyor...")
    
    while cap.isOpened(): # Eğer kamera açık ise
        ret, frame = cap.read() # ret = görüntü okundu mu, frame = görüntü
        if not ret: # Eğer görüntü okunamadıysa
            break
        
        # Vücut pose tespiti
        pose_results = detect_pose(frame)  
        draw_pose(frame, pose_results)
        
        # El tespiti
        hands_data = detect_hands(frame)
        draw_hands(frame, hands_data)
        
        # Sayı tanıma
        current_number = 0 # current_number = sayı
        
        # 0 tespiti - hiç el yoksa veya tüm parmaklar kapalıysa
        if len(hands_data) == 0:
            current_number = 0
        else:
            # Tek el sayı tanıma (0-5)
            for hand in hands_data:
                number = recognize_number(hand['landmarks'])
                if number >= 0:  # 0 dahil
                    current_number = number
                    break
            
            # İki el sayı tanıma (6-10)
            if len(hands_data) >= 2:
                dual_number = recognize_dual_hand_number(hands_data)
                if dual_number > 0:
                    current_number = dual_number
        
        # Sayıyı göster
        draw_number(frame, current_number)
        
        # FPS göster
        update_fps()
        cv2.putText(frame, f"FPS: {current_fps}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Frame'i göster
        cv2.imshow('MediaPipe Simple System - Mac M4', frame) # cv2.imshow = görüntüyü göster
        
        # ESC tuşu ile çıkış
        if cv2.waitKey(5) & 0xFF == 27: # cv2.waitKey = tuşa basıldı mı, 0xFF = 255, 27 = ESC tuşu
            break
    
    # Temizlik
    cap.release() # cap.release = kamera kapat
    cv2.destroyAllWindows() # cv2.destroyAllWindows = tüm pencereleri kapat
    print("✅ MediaPipe basit sistemi kapatıldı!")

if __name__ == "__main__": # __name__ = modül adı, "__main__" = ana modül
    try:
        main() # main = ana fonksiyon
    except KeyboardInterrupt:
        print("\n⏹️ Sistem kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"❌ Hata: {e}")

# =============================================================================
# 📜 LICENSE AND COPYRIGHT INFORMATION
# =============================================================================
"""
🤖 MediaPipe Hand Tracking & Finger Counting System
👨‍💻 Developer: Polat Sakarya
📅 Date: 2025

⚠️  COPYRIGHT WARNING:
This software is developed by Polat Sakarya and all rights are reserved. 
Unauthorized copying, distribution, modification, or commercial use of this code 
is strictly prohibited.

🔒 PROTECTION:
- This code is shared for educational purposes only
- Written permission required for commercial use
- Modification of source code is prohibited
- Unauthorized distribution is prohibited

📞 Contact: polatsakarya35@gmail.com

© 2025 Polat Sakarya - All Rights Reserved
"""
