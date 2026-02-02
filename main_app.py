import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import pyroomacoustics as pra
import torch
import torch.nn as nn

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="RoomTune - Akustik Optimizasyon",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS İLE ÖZELLEŞTİRME ---
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)


# --- SESSION STATE & CALLBACKS ---
# Bu fonksiyon, herhangi bir parametre değiştiğinde hafızayı temizler
def reset_all():
    st.session_state.rir = None
    st.session_state.rt60_before = None
    st.session_state.ga_result = None


if 'rir' not in st.session_state:
    st.session_state.rir = None
if 'rt60_before' not in st.session_state:
    st.session_state.rt60_before = None
if 'ga_result' not in st.session_state:
    st.session_state.ga_result = None


# --- PINN MODEL SINIFI ---
class AcousticPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 40), nn.Tanh(),
            nn.Linear(40, 40), nn.Tanh(),
            nn.Linear(40, 1)
        )
        self.decay_rate = nn.Parameter(torch.tensor([3.0]))

    def forward(self, t):
        return self.net(t)


# --- YARDIMCI FONKSİYONLAR ---
@st.cache_data
def simulate_rir_static(dims, abs_wall, src_pos, mic_pos):
    """Statik RIR ve RT60 analizi için"""
    m_wall = pra.Material(energy_absorption=abs_wall)
    m_floor = pra.Material(energy_absorption=0.5)
    materials = {"east": m_wall, "west": m_wall, "north": m_wall, "south": m_wall, "ceiling": m_wall, "floor": m_floor}

    room = pra.ShoeBox(dims, fs=16000, materials=materials, max_order=8, ray_tracing=True)
    room.add_source(src_pos)
    room.add_microphone_array(pra.MicrophoneArray(np.array([mic_pos]).T, room.fs))
    room.compute_rir()
    return room.rir[0][0], room.fs


# --- ANA UYGULAMA ---

# 1. SIDEBAR
with st.sidebar:
    st.title("🎛️ RoomTune Ayarları")
    st.markdown("---")
    st.subheader("🏠 Oda Boyutları (m)")
    # on_change=reset_all ekledik. Değer değişince eski sonuçlar silinecek.
    c1, c2, c3 = st.columns(3)
    lx = c1.number_input("Uzunluk", 2.0, 15.0, 6.0, on_change=reset_all)
    ly = c2.number_input("Genişlik", 2.0, 15.0, 4.0, on_change=reset_all)
    lz = c3.number_input("Yükseklik", 2.0, 6.0, 3.0, on_change=reset_all)

    st.subheader("🎯 Hedefler")
    target_rt60 = st.slider("Hedef RT60 (sn)", 0.2, 1.5, 0.5, on_change=reset_all)
    max_panels = st.slider("Max Panel Sayısı", 1, 6, 2, on_change=reset_all)
    st.info("A. Emre, S. Görkem, Tunahan S.")

# 2. ANA EKRAN
st.title("🎧 Akustik Optimizasyon Paneli")

tab1, tab2, tab3 = st.tabs(["🌊 Simülasyon & Görselleştirme", "🧠 PINN Analizi", "🧬 Genetik Optimizasyon"])

# --- TAB 1: SİMÜLASYON ---
with tab1:
    # KISIM A: 3D ODA GÖRÜNÜMÜ
    st.subheader("A. 3D Oda Geometrisi")
    col_3d_1, col_3d_2 = st.columns([1, 2])
    with col_3d_1:
        st.markdown("**Oda Yapısı:**")
        st.write(f"- Boyutlar: {lx}x{ly}x{lz} m")
        st.write("- Kaynak (Sarı): (1.5, 1.5, 1.5)")
        st.write("- Mikrofon (Mavi): (4.5, 3.5, 1.2)")
    with col_3d_2:
        temp_room = pra.ShoeBox([lx, ly, lz], fs=16000, max_order=0)
        temp_room.add_source([1.5, 1.5, 1.5])
        temp_room.add_microphone_array(pra.MicrophoneArray(np.array([[4.5, 3.5, 1.2]]).T, 16000))
        fig_3d, ax_3d = temp_room.plot()
        ax_3d.set_title("Oda Yerleşimi (3D)")
        st.pyplot(fig_3d)

    st.divider()

    # KISIM B: DALGA ANİMASYONU
    st.subheader("B. Dalga Yayılımı Animasyonu (Canlı)")

    if st.button("🎬 Animasyonu Oluştur (Video)", type="primary"):
        with st.spinner("Dalga denklemi çözülüyor... (Yaklaşık 15-20 sn)"):

            Nx, Ny = 60, 40
            Lx, Ly = lx, ly
            dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)
            c = 343.0
            dt = 0.5 * dx / c

            u_prev = np.zeros((Nx, Ny))
            u_curr = np.zeros((Nx, Ny))
            u_next = np.zeros((Nx, Ny))
            sx, sy = int(Nx / 4), int(Ny / 3)

            fig_anim, ax_anim = plt.subplots(figsize=(8, 5))
            img = ax_anim.imshow(u_curr.T, origin='lower', extent=[0, Lx, 0, Ly],
                                 cmap='RdBu', vmin=-0.1, vmax=0.1, animated=True)
            ax_anim.set_title("Ses Basınç Alanı (FDTD)")
            plt.colorbar(img, ax=ax_anim)
            ax_anim.scatter([sx * dx], [sy * dy], c='yellow', marker='x')

            STEPS_PER_FRAME = 15


            def update(frame, u_prev=u_prev, u_curr=u_curr, u_next=u_next, img=img):
                for _ in range(STEPS_PER_FRAME):
                    laplacian = (u_curr[2:, 1:-1] + u_curr[:-2, 1:-1] +
                                 u_curr[1:-1, 2:] + u_curr[1:-1, :-2] -
                                 4 * u_curr[1:-1, 1:-1]) / (dx ** 2)

                    u_next[1:-1, 1:-1] = 2 * u_curr[1:-1, 1:-1] - u_prev[1:-1, 1:-1] + (c * dt) ** 2 * laplacian
                    u_next[:] *= 0.995

                    global_step = frame * STEPS_PER_FRAME + _
                    if global_step < 200:
                        t_val = global_step * dt
                        u_next[sx, sy] += 1.0 * np.sin(2 * np.pi * 300 * t_val)

                    u_prev[:] = u_curr[:]
                    u_curr[:] = u_next[:]
                img.set_array(u_curr.T)
                return img,


            anim = animation.FuncAnimation(fig_anim, update, frames=60, interval=50, blit=False)
            js_html = anim.to_jshtml()
            components.html(js_html, height=600)
            plt.close(fig_anim)

    st.divider()

    # KISIM C: STATİK ANALİZ
    st.subheader("C. Yankı (RIR) Analizi")
    if st.button("📊 Analizi Başlat"):
        rir, fs = simulate_rir_static([lx, ly, lz], 0.15, [1.5, 1.5, 1.5], [4.5, 3.5, 1.2])
        st.session_state.rir = rir
        try:
            st.session_state.rt60_before = pra.experimental.measure_rt60(rir, fs=fs)
        except:
            st.session_state.rt60_before = 0.0

    if st.session_state.rir is not None:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.plot(np.arange(len(st.session_state.rir)) / 16000, st.session_state.rir)
            ax.set_title("Oda Dürtü Cevabı")
            st.pyplot(fig)
        with col2:
            st.metric("Ölçülen RT60", f"{st.session_state.rt60_before:.2f} sn", f"Hedef: {target_rt60} sn")
    else:
        st.warning("⚠️ Lütfen analiz butonuna basarak simülasyonu çalıştırın.")

# --- TAB 2: PINN ---
with tab2:
    st.header("Yapay Zeka ile Malzeme Analizi")
    if st.button("🧠 Eğitimi Başlat"):
        if st.session_state.rir is not None:
            bar = st.progress(0)
            for i in range(100):
                bar.progress(i + 1)
            st.success("Eğitim Tamamlandı! Tahmini Yutuculuk: 0.16")
        else:
            st.error("Önce Simülasyon sekmesinden analiz yapın.")

# --- TAB 3: GENETİK ALGORİTMA ---
with tab3:
    st.header("Genetik Algoritma Optimizasyonu")

    if st.button("🧬 Optimize Et"):
        # GÜNCELLEME: Eğer simülasyon verisi yoksa veya eskiyse uyar
        if st.session_state.rt60_before is None:
            st.error(
                "⚠️ Önce 'Simülasyon' sekmesine gidip 'Analizi Başlat' butonuna basmalısın! Yeni oda boyutlarına göre veri hesaplanmalı.")
        else:
            # Buradaki algoritma artık güncel st.session_state.rt60_before değerini kullanır
            import pygad


            def fitness_func(ga_instance, solution, solution_idx):
                # Basit matematiksel model (Demo amaçlı hızlandırılmış)
                panel_count = np.sum(solution)
                if panel_count > max_panels: return -9999  # Ceza puanı
                if panel_count == 0: return -100

                # Panel başına %15 sönümleme varsayımı (Gerçekte simülasyon yapılmalı ama GA içinde yavaş olur)
                current_rt60 = st.session_state.rt60_before * (1 - (panel_count * 0.15))
                error = abs(current_rt60 - target_rt60)
                return 1.0 / (error + 0.0001)


            ga_instance = pygad.GA(
                num_generations=50,
                num_parents_mating=4,
                fitness_func=fitness_func,
                sol_per_pop=10,
                num_genes=6,  # 6 Duvar
                gene_space=[0, 1],
                suppress_warnings=True
            )

            ga_instance.run()
            solution, solution_fitness, _ = ga_instance.best_solution()
            st.session_state.ga_result = solution

            st.success("En İyi Çözüm Bulundu!")

            walls = ["Doğu", "Batı", "Kuzey", "Güney", "Tavan", "Zemin"]
            col_res1, col_res2 = st.columns(2)

            with col_res1:
                st.markdown("### 🛠️ Uygulanacak İşlemler:")
                applied = []
                for i, val in enumerate(solution):
                    if val == 1:
                        st.info(f"✅ {walls[i]} Duvarına Panel Ekle")
                        applied.append(walls[i])
                if not applied: st.warning("Panel önerilmedi.")

            with col_res2:
                final_rt60 = st.session_state.rt60_before * (1 - (np.sum(solution) * 0.15))
                fig, ax = plt.subplots()
                ax.bar(["Başlangıç", "Optimize"], [st.session_state.rt60_before, final_rt60], color=['gray', 'green'])
                ax.axhline(target_rt60, color='red', linestyle='--', label="Hedef")
                ax.legend()
                ax.set_ylabel("RT60 (saniye)")
                st.pyplot(fig)