import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# 1. CẤU HÌNH TRANG & DỮ LIỆU TRA CỨU TIÊU CHUẨN
# ==============================================================================
st.set_page_config(
    page_title="Phần mềm Tính toán Thấm Thủy Công - TLU",
    page_icon="🌊",
    layout="wide"
)

BANG_HE_SO_C_LANE = {
    "Đất sét chặt (C = 1.5)": 1.5,
    "Đất sét chặt vừa (C = 1.85)": 1.85,
    "Đất sét mềm, đất thịt (C = 2.25)": 2.25,
    "Đá cuội, sỏi hạt lớn (C = 2.5)": 2.5,
    "Cuội sỏi hạt trung bình (C = 3.0)": 3.0,
    "Cuội sỏi hạt nhỏ (C = 3.5)": 3.5,
    "Cát hạt lớn (C = 4.0)": 4.0,
    "Cát hạt trung bình (C = 5.0)": 5.0,
    "Cát hạt nhỏ (C = 6.0)": 6.0,
    "Cát mịn (C = 7.0)": 7.0,
}

# ==============================================================================
# 2. CÁC HÀM LÝ THUYẾT THẤM (HỆ SỐ SỨC KHÁNG & TỶ LỆ ĐƯỜNG THẲNG)
# ==============================================================================
def tinh_chieu_day_tang_tham_tt(L0, S0, T0):
    """Xác định chiều dày tầng thấm tính toán T_tt theo Bảng 3-2 (Chugaev)"""
    ty_so = L0 / S0 if S0 > 0 else 999.0
    if ty_so > 5.0:
        alpha = 0.5
        T_tt = 0.5 * L0
        ct_str = f"Do \\frac{{L_0}}{{S_0}} = \\frac{{{L0}}}{{{S0}}} = {ty_so:.2f} > 5 \\Rightarrow \\alpha = 0,5 \\Rightarrow T_{{tt}} = 0,5 L_0 = 0,5 \\times {L0} = {T_tt:.2f}\\text{{ m}}"
    else:
        alpha = 0.85 - 0.07 * ty_so
        T_tt = alpha * L0 + S0
        ct_str = f"Do \\frac{{L_0}}{{S_0}} = {ty_so:.2f} \\le 5 \\Rightarrow \\alpha = 0,85 - 0,07\\frac{{L_0}}{{S_0}} = {alpha:.4f} \\Rightarrow T_{{tt}} = \\alpha L_0 + S_0 = {T_tt:.2f}\\text{{ m}}"
    
    if T0 <= T_tt:
        T_chon = T0
        kl_str = f"Vì T_0 = {T0}\\text{{ m}} \\le T_{{tt}} = {T_tt:.2f}\\text{{ m}} nên lấy chiều dày tầng thấm thực tế T = T_0 = {T0}\\text{{ m}} để tính toán."
    else:
        T_chon = T_tt
        kl_str = f"Vì T_0 = {T0}\\text{{ m}} > T_{{tt}} = {T_tt:.2f}\\text{{ m}} nên lấy chiều dày tầng thấm tính toán T = T_{{tt}} = {T_tt:.2f}\\text{{ m}} để tính toán."
    return T_tt, T_chon, ty_so, alpha, ct_str, kl_str

def tinh_xi_cua_vao_ra(a, S, T):
    """Tính hệ số sức kháng bộ phận cửa vào hoặc cửa ra: xi = 0.44 + xi_b + xi_c"""
    xi_b = 1.05 * (a / T) + 0.4 * ((a / T) ** 2) if (a > 0 and T > 0) else 0.0
    T1 = max(T - a, 0.001)
    xi_c = 0.64 * (S / T1) + 1.5 * ((S / T1) ** 2) if (S > 0 and T1 > 0) else 0.0
    xi_tong = 0.44 + xi_b + xi_c
    ct = f"\\xi_b = {xi_b:.4f}, \\xi_c = {xi_c:.4f} (với T_1 = T - a = {T1:.2f}m) \\Rightarrow \\xi = 0,44 + {xi_b:.4f} + {xi_c:.4f} = {xi_tong:.4f}"
    return xi_tong, xi_b, xi_c, T1, ct

def tinh_xi_bo_phan_giua(a, S, T_truoc, T_sau):
    """Tính hệ số sức kháng bộ phận giữa thứ k"""
    T_max = max(T_truoc, T_sau, 0.001)
    T_min = max(min(T_truoc, T_sau), 0.001)
    delta_T = abs(T_truoc - T_sau) if a == 0 else a
    
    if delta_T == 0:
        xi_b = 0.0
        xi_c = 1.28 * (S / T_max) + 3.0 * ((S / T_max) ** 2) if S > 0 else 0.0
        ct = f"Không bậc (a=0), có cừ S={S}m, T={T_max}m: \\xi_g = 1,28\\frac{{S}}{{T}} + 3,0\\left(\\frac{{S}}{{T}}\\right)^2 = {xi_c:.4f}"
    else:
        xi_b = 0.72 * (delta_T / T_max) + 0.30 * ((delta_T / T_max) ** 2)
        xi_c = 1.28 * (S / T_min) + 3.0 * ((S / T_min) ** 2) if S > 0 else 0.0
        ct = f"Có bậc a={delta_T}m (\\xi_b={xi_b:.4f}) và cừ S={S}m (\\xi_c={xi_c:.4f}) \\Rightarrow \\xi_g = \\xi_b + \\xi_c = {xi_b+xi_c:.4f}"
    return xi_b + xi_c, xi_b, xi_c, ct

def tinh_xi_nam_ngang(l, T_ngang, S_trai, S_phai):
    """Tính hệ số sức kháng bộ phận nằm ngang có kiểm tra điều kiện l >= 0.5*(S' + S'')"""
    dk = 0.5 * (S_trai + S_phai)
    if T_ngang <= 0:
        return 0.0, "T <= 0"
    if l >= dk:
        xi_n = l / T_ngang
        ct = f"Thỏa mãn l = {l}m \\ge 0,5(S' + S'') = {dk:.2f}m \\Rightarrow \\xi_n = \\frac{{l}}{{T}} = \\frac{{{l}}}{{{T_ngang}}} = {xi_n:.4f}"
    else:
        # Công thức hiệu chỉnh khi 2 hàng cừ quá gần nhau: l < 0.5*(S' + S'')
        xi_n = (l / T_ngang) * np.sqrt(l / dk) if dk > 0 else l / T_ngang
        ct = f"Do l = {l}m < 0,5(S' + S'') = {dk:.2f}m (cừ gần nhau) \\Rightarrow \\xi_n = {xi_n:.4f}"
    return xi_n, ct

# ==============================================================================
# 3. HÀM VẼ ĐỒ THỊ MẶT CẮT, ĐƯỜNG ĐO ÁP & BIỂU ĐỒ ÁP LỰC ĐẨY NGƯỢC
# ==============================================================================
def ve_bieu_do_ap_luc_day_nguoc(L_AB, h_A, h_B, h_t, gamma=10.0):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.array([0, L_AB])
    p_tt = np.array([gamma * h_t, gamma * h_t])
    p_th = np.array([gamma * h_A, gamma * h_B])
    p_tong = p_tt + p_th

    ax.plot([0, L_AB], [0, 0], color='#1E293B', linewidth=4, label='Bản đáy công trình (AB)')
    ax.fill_between(x, 0, -p_tt, color='#38BDF8', alpha=0.45, label=f'Áp lực thủy tĩnh W_tt (γ.h_t = {gamma*h_t:.1f} kPa)')
    ax.fill_between(x, -p_tt, -p_tong, color='#FB923C', alpha=0.55, label='Áp lực thấm W_th (γ.h_x)')
    
    ax.plot(x, -p_tt, color='#0284C7', linestyle='--', linewidth=1.5)
    ax.plot(x, -p_tong, color='#EA580C', linewidth=2.5, marker='o')

    ax.annotate(f"Đầu đáy A\nγ.h_t = {p_tt[0]:.1f}\nγ.h_A = {p_th[0]:.2f}\nTổng = {p_tong[0]:.2f} kPa", 
                xy=(0, -p_tong[0]), xytext=(0.04*L_AB, -p_tong[0]*0.92),
                fontsize=8.5, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#EA580C", alpha=0.9))
    
    ax.annotate(f"Cuối đáy B\nγ.h_t = {p_tt[1]:.1f}\nγ.h_B = {p_th[1]:.2f}\nTổng = {p_tong[1]:.2f} kPa", 
                xy=(L_AB, -p_tong[1]), xytext=(0.68*L_AB, -p_tong[1]*1.05),
                fontsize=8.5, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#EA580C", alpha=0.9))

    ax.set_title("BIỂU ĐỒ ÁP LỰC ĐẨY NGƯỢC DƯỚI BẢN ĐÁY (AB)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Chiều dài dọc bản đáy AB (m)")
    ax.set_ylabel("Áp lực đẩy ngược (kPa = kN/m²)")
    ax.set_xlim(-0.08 * L_AB, 1.08 * L_AB)
    ax.set_ylim(-max(p_tong) * 1.25, max(p_tong) * 0.15)
    
    ticks = ax.get_yticks()
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{abs(t):.0f}" if t <= 0 else "" for t in ticks])
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower left', fontsize=8.5)
    plt.tight_layout()
    return fig

def ve_duong_do_ap_doc_duong_vien(df_ket_qua, H):
    """Vẽ biểu đồ phân bố cột nước thấm h(x) dọc theo các bộ phận từ Thượng lưu về Hạ lưu"""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ten_bp = ["Thượng lưu"] + df_ket_qua["Bộ phận"].tolist()
    cot_nuoc = [H] + df_ket_qua["Cột nước còn lại (m)"].tolist()
    x_idx = np.arange(len(ten_bp))

    ax.plot(x_idx, cot_nuoc, marker='s', color='#D97706', linewidth=2.5, markersize=7, label='Đường cột nước thấm h(x)')
    ax.fill_between(x_idx, 0, cot_nuoc, color='#FDE68A', alpha=0.45)

    for i, txt in enumerate(cot_nuoc):
        ax.annotate(f"{txt:.3f} m", (x_idx[i], cot_nuoc[i]), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8.5, fontweight='bold')

    ax.set_title("SỰ SUY GIẢM CỘT NƯỚC THẤM QUA TỪNG BỘ PHẬN", fontsize=11, fontweight='bold')
    ax.set_xticks(x_idx)
    ax.set_xticklabels(ten_bp, rotation=20, ha='right', fontsize=8.5)
    ax.set_ylabel("Cột nước thấm h (m)")
    ax.set_ylim(0, H * 1.18)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')
    plt.tight_layout()
    return fig

# ==============================================================================
# 4. GIAO DIỆN CHÍNH & SIDEBAR ĐIỀU KHIỂN
# ==============================================================================
st.title("🌊 HỆ THỐNG TÍNH TOÁN THẤM CÔNG TRÌNH THỦY LỢI (CHUGAEV & BLIGH-LANE)")
st.markdown("""
*Công cụ chuyên dụng phục vụ tính toán thiết kế, giảng dạy và tự động xuất thuyết minh tính toán thấm dưới đáy công trình thủy lợi.*
""")

st.sidebar.header("⚙️ LỰA CHỌN PHƯƠNG PHÁP")
phuong_phap = st.sidebar.radio(
    "Phương pháp tính toán:",
    [
        "1. Phương pháp Hệ số sức kháng (Chugaev)",
        "2. Phương pháp Tỷ lệ đường thẳng (Bligh - Lane)"
    ]
)

st.sidebar.divider()
st.sidebar.subheader("📂 Nạp bài tập mẫu (Slide TLU)")

# ==============================================================================
# NHÁNH 1: PHƯƠNG PHÁP HỆ SỐ SỨC KHÁNG (HỖ TRỢ N ĐOẠN NGANG & N-1 BỘ PHẬN GIỮA)
# ==============================================================================
if "Hệ số sức kháng" in phuong_phap:
    preset = st.sidebar.selectbox(
        "Chọn dữ liệu mẫu:",
        [
            "Ví dụ Cống B (Slide 24-35)",
            "Ví dụ Cống C (Slide 35-42)",
            "Tùy chỉnh N hàng cừ bất kỳ (Dynamic)"
        ]
    )

    if preset == "Ví dụ Cống C (Slide 35-42)":
        def_H1, def_H2, def_L0, def_S0, def_T0 = 14.0, 2.0, 35.0, 6.5, 17.5
        def_a_vao, def_S_vao = 0.5, 0.0
        def_a_ra, def_S_ra = 0.5, 3.0
        def_so_doan_ngang = 2
        def_ngang = [{"l": 20.0, "T": 17.0}, {"l": 15.0, "T": 17.0}]
        def_giua = [{"a": 0.0, "S": 7.5}]
        def_vi_tri_day = 2
        def_LAB, def_ht = 15.0, 2.5
        def_k, def_Jtb_gh, def_Jra_gh = 0.0002, 0.25, 0.45
    else:
        def_H1, def_H2, def_L0, def_S0, def_T0 = 11.0, 3.0, 32.0, 5.0, 11.0
        def_a_vao, def_S_vao = 0.5, 0.0
        def_a_ra, def_S_ra = 1.0, 1.0
        def_so_doan_ngang = 2
        def_ngang = [{"l": 20.0, "T": 11.0}, {"l": 12.0, "T": 10.0}]
        def_giua = [{"a": 1.0, "S": 4.0}]
        def_vi_tri_day = 2
        def_LAB, def_ht = 12.0, 4.0
        def_k, def_Jtb_gh, def_Jra_gh = 0.0002, 0.29, 0.44

    st.subheader("1️⃣ Thông số thủy lực, tầng thấm và bản đáy công trình")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        H1 = st.number_input("Cột nước thượng lưu H₁ (m)", value=def_H1, step=0.5)
        H2 = st.number_input("Cột nước hạ lưu H₂ (m)", value=def_H2, step=0.5)
    with col2:
        L0 = st.number_input("Hình chiếu ngang đường viền L₀ (m)", value=def_L0, step=1.0)
        S0 = st.number_input("Hình chiếu đứng đường viền S₀ (m)", value=def_S0, step=0.5)
    with col3:
        T0 = st.number_input("Chiều dày tầng thấm thực tế T₀ (m)", value=def_T0, step=0.5)
        L_AB = st.number_input("Chiều dài bản đáy tính áp lực L_AB (m)", value=def_LAB, step=0.5)
    with col4:
        h_t = st.number_input("Chiều sâu ngập bản đáy hạ lưu h_t (m)", value=def_ht, step=0.5)
        gamma = st.number_input("Trọng lượng riêng nước γ (kN/m³)", value=10.0, step=0.1)

    H = H1 - H2
    T_tt, T_chon, ty_so_LS, alpha_T, ct_T, kl_T = tinh_chieu_day_tang_tham_tt(L0, S0, T0)

    st.subheader("2️⃣ Cấu hình phân đoạn miền thấm (Hỗ trợ N đoạn nằm ngang & N-1 bộ phận giữa)")
    c_cfg1, c_cfg2, c_cfg3 = st.columns([1, 1, 1.5])
    with c_cfg1:
        so_doan_ngang = st.number_input("Số bộ phận nằm ngang (N ≥ 1)", min_value=1, max_value=6, value=def_so_doan_ngang, step=1)
    with c_cfg2:
        doan_ban_day = st.selectbox(
            "Bộ phận nằm ngang nào là Bản đáy AB?",
            options=list(range(1, so_doan_ngang + 1)),
            index=min(def_vi_tri_day - 1, so_doan_ngang - 1),
            format_func=lambda x: f"Đoạn nằm ngang thứ {x} (Từ trái sang phải)"
        )
    with c_cfg3:
        use_slide_round = st.checkbox(
            "Áp dụng làm tròn trung gian giống Slide Cống B (ξ = 0.489; 1.818; 1.403; 1.200; 0.683)",
            value=(preset == "Ví dụ Cống B (Slide 24-35)")
        )

    # Nhập Cửa vào & Cửa ra
    col_in, col_mid, col_out = st.columns([1, 2, 1])
    with col_in:
        st.markdown("##### 🔹 Bộ phận Cửa vào (Đầu tiên)")
        a_vao = st.number_input("Chiều sâu bậc cửa vào a_vào (m)", value=def_a_vao, step=0.1)
        S_vao = st.number_input("Chiều sâu cừ cửa vào S_vào (m)", value=def_S_vao, step=0.5)

    with col_out:
        st.markdown("##### 🔹 Bộ phận Cửa ra (Cuối cùng)")
        a_ra = st.number_input("Chiều sâu bậc cửa ra a_ra (m)", value=def_a_ra, step=0.1)
        S_ra = st.number_input("Chiều sâu cừ cửa ra S_ra (m)", value=def_S_ra, step=0.5)

    with col_mid:
        st.markdown("##### 🔹 Các đoạn nằm ngang & bộ phận giữa xen kẽ")
        ds_ngang_input = []
        ds_giua_input = []
        for i in range(so_doan_ngang):
            default_l = def_ngang[i]["l"] if i < len(def_ngang) else 10.0
            default_T = def_ngang[i]["T"] if i < len(def_ngang) else T_chon
            cn1, cn2 = st.columns(2)
            l_val = cn1.number_input(f"Chiều dài đoạn ngang {i+1}: l_n{i+1} (m)", value=default_l, step=1.0, key=f"ln_{i}")
            T_val = cn2.number_input(f"Tầng thấm dưới đoạn ngang {i+1}: T_n{i+1} (m)", value=default_T, step=0.5, key=f"Tn_{i}")
            ds_ngang_input.append({"l": l_val, "T": T_val})

            if i < so_doan_ngang - 1:
                default_ag = def_giua[i]["a"] if i < len(def_giua) else 0.0
                default_Sg = def_giua[i]["S"] if i < len(def_giua) else 4.0
                cg1, cg2 = st.columns(2)
                a_g = cg1.number_input(f"↳ Bậc giữa {i+1} (ngăn cách ngang {i+1} & {i+2}): a_g{i+1} (m)", value=default_ag, step=0.1, key=f"ag_{i}")
                S_g = cg2.number_input(f"↳ Cừ giữa {i+1} (ngăn cách ngang {i+1} & {i+2}): S_g{i+1} (m)", value=default_Sg, step=0.5, key=f"Sg_{i}")
                ds_giua_input.append({"a": a_g, "S": S_g})

    # Giải bài toán tuần tự qua từng bộ phận
    cac_bo_phan = []
    
    # 1. Cửa vào
    xi_v, xi_bv, xi_cv, T1_v, ct_v = tinh_xi_cua_vao_ra(a_vao, S_vao, T_chon)
    cac_bo_phan.append({
        "ten": "(1) Cửa vào",
        "loai": "vao",
        "mo_ta": f"a={a_vao}m, S={S_vao}m, T={T_chon:.2f}m",
        "xi": xi_v,
        "cong_thuc": ct_v
    })

    # 2. Duyệt xen kẽ Ngang -> Giữa
    stt = 2
    idx_bo_phan_truoc_A = 0
    idx_bo_phan_tai_B = 0

    for i in range(so_doan_ngang):
        # Xác định chiều sâu cừ/bậc 2 đầu đoạn ngang i để kiểm tra l >= 0.5*(S' + S'')
        S_trai = (a_vao + S_vao) if i == 0 else (ds_giua_input[i-1]["a"] + ds_giua_input[i-1]["S"])
        S_phai = (a_ra + S_ra) if i == (so_doan_ngang - 1) else (ds_giua_input[i]["a"] + ds_giua_input[i]["S"])
        
        l_cur = ds_ngang_input[i]["l"]
        T_cur = ds_ngang_input[i]["T"]
        xi_n, ct_n = tinh_xi_nam_ngang(l_cur, T_cur, S_trai, S_phai)
        
        la_ban_day = (i + 1 == doan_ban_day)
        ten_ngang = f"({stt}) Nằm ngang {i+1}" + (" [Bản đáy AB]" if la_ban_day else "")
        if la_ban_day:
            idx_bo_phan_truoc_A = len(cac_bo_phan)  # Điểm A nằm ngay trước đoạn ngang bản đáy
            idx_bo_phan_tai_B = len(cac_bo_phan) + 1 # Điểm B nằm ngay sau đoạn ngang bản đáy

        cac_bo_phan.append({
            "ten": ten_ngang,
            "loai": "ngang",
            "mo_ta": f"l={l_cur}m, T={T_cur}m",
            "xi": xi_n,
            "cong_thuc": ct_n
        })
        stt += 1

        # Nếu chưa phải đoạn ngang cuối thì thêm bộ phận giữa
        if i < so_doan_ngang - 1:
            a_g = ds_giua_input[i]["a"]
            S_g = ds_giua_input[i]["S"]
            T_tr = ds_ngang_input[i]["T"]
            T_sa = ds_ngang_input[i+1]["T"]
            xi_g, _, _, ct_g = tinh_xi_bo_phan_giua(a_g, S_g, T_tr, T_sa)
            cac_bo_phan.append({
                "ten": f"({stt}) Bộ phận giữa {i+1}",
                "loai": "giua",
                "mo_ta": f"a={a_g}m, S={S_g}m",
                "xi": xi_g,
                "cong_thuc": ct_g
            })
            stt += 1

    # 3. Cửa ra
    xi_r, xi_br, xi_cr, T1_r, ct_r = tinh_xi_cua_vao_ra(a_ra, S_ra, T_chon)
    cac_bo_phan.append({
        "ten": f"({stt}) Cửa ra",
        "loai": "ra",
        "mo_ta": f"a={a_ra}m, S={S_ra}m, T={T_chon:.2f}m",
        "xi": xi_r,
        "cong_thuc": ct_r
    })

    # Áp dụng làm tròn mẫu Cống B nếu người dùng bật checkbox
    if use_slide_round and preset == "Ví dụ Cống B (Slide 24-35)" and len(cac_bo_phan) == 5:
         xi_mau = [0.489, 1.818, 1.403, 1.200, 0.683]
         for idx_m, val_m in enumerate(xi_mau):
             cac_bo_phan[idx_m]["xi"] = val_m

    # Tổng hợp kết quả
    tong_xi = sum(bp["xi"] for bp in cac_bo_phan)
    h_con_lai = H
    for bp in cac_bo_phan:
        bp["h_i"] = H * (bp["xi"] / tong_xi) if tong_xi > 0 else 0.0
        h_con_lai -= bp["h_i"]
        bp["h_con_lai"] = max(h_con_lai, 0.0)

    # Cột nước tại A (ngay trước đoạn ngang bản đáy) và tại B (ngay sau đoạn ngang bản đáy)
    h_A = H - sum(cac_bo_phan[k]["h_i"] for k in range(idx_bo_phan_truoc_A))
    h_B = H - sum(cac_bo_phan[k]["h_i"] for k in range(idx_bo_phan_tai_B))

    W_th = 0.5 * (h_A + h_B) * gamma * L_AB
    W_tt = gamma * h_t * L_AB
    W_dn = W_th + W_tt

    # Thông số kiểm tra độ bền & lưu lượng
    st.subheader("3️⃣ Thông số kiểm tra độ bền thấm & Lưu lượng")
    k1, k2, k3, k4 = st.columns(4)
    k_val = k1.number_input("Hệ số thấm của nền k (m/s)", value=def_k, format="%.5f")
    J_tb_gh = k2.number_input("Gradien chung cho phép [J_tb]", value=def_Jtb_gh, step=0.01)
    J_ra_gh = k3.number_input("Gradien cửa ra cho phép [J_ra]", value=def_Jra_gh, step=0.01)
    alpha_ra = k4.number_input("Hệ số cửa ra α_ra (tra theo S_ra/T)", value=0.66, step=0.01)

    J_tb = H / L0 if L0 > 0 else 0.0
    S_ra_tong = max(a_ra + S_ra, 0.001)
    J_ra = (H / (tong_xi * S_ra_tong)) * alpha_ra if tong_xi > 0 else 0.0
    q_tham = k_val * (H / tong_xi) if tong_xi > 0 else 0.0

    # Hiển thị Tabs kết quả
    tab1, tab2, tab3 = st.tabs([
        "📊 Bảng Kết Quả & Biểu Đồ Trực Quan",
        "📝 Xuất Thuyết Minh Chi Tiết (Từng Bước Thay Số)",
        "📖 Tóm Tắt Công Thức Lý Thuyết"
    ])

    with tab1:
        df_show = pd.DataFrame({
            "Bộ phận": [bp["ten"] for bp in cac_bo_phan],
            "Thông số": [bp["mo_ta"] for bp in cac_bo_phan],
            "Hệ số sức kháng ξ_i": [round(bp["xi"], 4) for bp in cac_bo_phan],
            "Tổn thất cột nước h_i (m)": [round(bp["h_i"], 3) for bp in cac_bo_phan],
            "Cột nước còn lại (m)": [round(bp["h_con_lai"], 3) for bp in cac_bo_phan]
        })
        st.dataframe(df_show, width="stretch")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tổng sức kháng ∑ξ", f"{tong_xi:.4f}")
        m2.metric("Cột nước tại A / B", f"{h_A:.3f}m / {h_B:.3f}m")
        m3.metric("Tổng áp lực đẩy ngược W_đn", f"{W_dn:.2f} kN/m", f"W_th={W_th:.2f} | W_tt={W_tt:.2f}")
        m4.metric("Lưu lượng thấm q", f"{q_tham:.6f} m³/s.m")

        ch1, ch2 = st.columns(2)
        with ch1:
            st.pyplot(ve_bieu_do_ap_luc_day_nguoc(L_AB, h_A, h_B, h_t, gamma))
        with ch2:
            st.pyplot(ve_duong_do_ap_doc_duong_vien(df_show, H))

    with tab2:
        # Tạo nội dung thuyết minh tự động bằng Markdown + LaTeX
        thuyet_minh = f"""# THUYẾT MINH TÍNH TOÁN THẤM DƯỚI ĐÁY CÔNG TRÌNH
**Phương pháp:** Hệ số sức kháng (Phương pháp Chugaev)

---

### 1. Xác định chiều dày tính toán của tầng thấm ($T_{{tt}}$)
- Chênh lệch cột nước thượng - hạ lưu: $H = H_1 - H_2 = {H1} - {H2} = {H:.2f}\\text{{ m}}$
- Hình chiếu ngang đường viền thấm: $L_0 = {L0}\\text{{ m}}$; Hình chiếu đứng: $S_0 = {S0}\\text{{ m}}$
- ${ct_T}$
- **Kết luận chọn tầng thấm:** ${kl_T}$

---

### 2. Tính hệ số sức kháng của từng bộ phận ($\\xi_i$)
"""
        for bp in cac_bo_phan:
            thuyet_minh += f"- **{bp['ten']}** ({bp['mo_ta']}):\n  $${bp['cong_thuc']}$$\n"

        thuyet_minh += f"""
- **Tổng hệ số sức kháng của toàn miền thấm:**
  $$\\sum \\xi = {" + ".join([f"{bp['xi']:.4f}" for bp in cac_bo_phan])} = \\mathbf{{{tong_xi:.4f}}}$$

---

### 3. Kiểm tra độ bền thấm chung và phân phối cột nước thấm
- **Gradien thấm chung của nền:**
  $$J_{{tb}} = \\frac{{H}}{{L_0}} = \\frac{{{H:.2f}}}{{{L0}}} = {J_tb:.3f} {"\\le" if J_tb <= J_tb_gh else ">"} [J_{{tb}}] = {J_tb_gh}$$
  $\\rightarrow$ Nền công trình **{"ĐẢM BẢO" if J_tb <= J_tb_gh else "KHÔNG ĐẢM BẢO"}** độ bền thấm chung.

- **Tổn thất cột nước qua từng bộ phận ($h_i = \\frac{{H}}{{\\sum \\xi}} \\xi_i$):**
"""
        for idx, bp in enumerate(cac_bo_phan, 1):
            thuyet_minh += f"  - $h_{idx} = \\frac{{{H:.2f}}}{{{tong_xi:.4f}}} \\times {bp['xi']:.4f} = \\mathbf{{{bp['h_i']:.3f}\\text{{ m}}}}$\n"

        thuyet_minh += f"""
---

### 4. Tính áp lực đẩy ngược tác dụng lên bản đáy công trình (AB)
- **Cột nước thấm tại đầu bản đáy A và cuối bản đáy B:**
  $$h_A = {h_A:.3f}\\text{{ m}} \\quad ; \\quad h_B = {h_B:.3f}\\text{{ m}}$$
- **Áp lực thấm tác dụng vào bản đáy AB ($L_{{AB}} = {L_AB}\\text{{ m}}$):**
  $$W_{{th}} = \\frac{{h_A + h_B}}{{2}} \\cdot \\gamma \\cdot L_{{AB}} = \\frac{{{h_A:.3f} + {h_B:.3f}}}{{2}} \\times {gamma} \\times {L_AB} = \\mathbf{{{W_th:.2f}\\text{{ kN/m}}}}$$
- **Áp lực thủy tĩnh đẩy ngược ($h_t = {h_t}\\text{{ m}}$):**
  $$W_{{tt}} = \\gamma \\cdot h_t \\cdot L_{{AB}} = {gamma} \\times {h_t} \\times {L_AB} = \\mathbf{{{W_tt:.2f}\\text{{ kN/m}}}}$$
- **Tổng áp lực đẩy ngược tác dụng lên bản đáy AB:**
  $$W_{{dn}} = W_{{th}} + W_{{tt}} = {W_th:.2f} + {W_tt:.2f} = \\mathbf{{{W_dn:.2f}\\text{{ kN/m}}}}$$

---

### 5. Kiểm tra độ bền thấm tại cửa ra và tính lưu lượng thấm
- **Gradien thấm lớn nhất tại cửa ra:**
  $$J_{{ra}} = \\frac{{H}}{{\\sum \\xi \\cdot S_{{ra}}}} \\alpha_{{ra}} = \\frac{{{H:.2f}}}{{{tong_xi:.4f} \\times {S_ra_tong:.2f}}} \\times {alpha_ra} = \\mathbf{{{J_ra:.3f}}} {"\\le" if J_ra <= J_ra_gh else ">"} [J_{{ra}}] = {J_ra_gh}$$
  $\\rightarrow$ Tại cửa ra **{"KHÔNG xảy ra xói ngầm cơ học" if J_ra <= J_ra_gh else "XẢY RA xói ngầm cơ học (Cần bố trí tầng lọc ngược / kéo dài đường viền)"}**.
- **Lưu lượng thấm đơn vị qua đáy công trình:**
  $$q = k \\frac{{H}}{{\\sum \\xi}} = {k_val:.5f} \\times \\frac{{{H:.2f}}}{{{tong_xi:.4f}}} = \\mathbf{{{q_tham:.6f}\\text{{ m}}^3/(\\text{{s}}\\cdot\\text{{m}})}}$$
"""
        st.markdown(thuyet_minh)
        st.download_button(
            label="📥 Tải Thuyết Minh Tính Toán (.md / Hỗ trợ chuyển sang Word/LaTeX)",
            data=thuyet_minh.encode("utf-8"),
            file_name="Thuyet_minh_tinh_tham_Chugaev.md",
            mime="text/markdown"
        )

    with tab3:
        st.markdown(r"""
        #### Tóm tắt công thức Phương pháp Hệ số sức kháng (Chugaev)
        1. **Bộ phận cửa vào, cửa ra:** $\xi_{v,r} = 0,44 + \xi_b + \xi_c$
           - $\xi_b = 1,05\left(\frac{a}{T}\right) + 0,4\left(\frac{a}{T}\right)^2$
           - $\xi_c = 0,64\left(\frac{S}{T_1}\right) + 1,5\left(\frac{S}{T_1}\right)^2$ với $T_1 = T - a$
        2. **Bộ phận giữa:** $\xi_g = \xi_b + \xi_c$
           - Khi không có bậc ($a = 0$): $\xi_g = 1,28\left(\frac{S}{T}\right) + 3,0\left(\frac{S}{T}\right)^2$
        3. **Bộ phận nằm ngang:** Nếu $l_i \ge 0,5(S' + S'') \Rightarrow \xi_{ni} = \frac{l_i}{T_i}$
        """)

# ==============================================================================
# NHÁNH 2: PHƯƠNG PHÁP TỶ LỆ ĐƯỜNG THẲNG (BLIGH - LANE)
# ==============================================================================
else:
    preset_lane = st.sidebar.selectbox(
        "Chọn dữ liệu mẫu:",
        ["Ví dụ Cống A (Slide 63-72)", "Ví dụ 4 (Slide 73)", "Tự nhập thông số"]
    )

    if preset_lane == "Ví dụ 4 (Slide 73)":
        def_H1, def_H2, def_m = 8.0, 3.0, 3.0
        def_ds_d = "4.0, 4.0, 4.0, 4.0, 5.0, 4.0"
        def_ds_n = "20.0, 9.0"
        def_Ld_A, def_Ln_A = 13.0, 9.0
        def_Ld_B, def_Ln_B = 4.0, 0.0
        def_LAB_th, def_LAB_tt, def_ht = 9.0, 9.0, 3.0
        def_d_ra, def_T1, def_T2 = 4.0, 11.0, 12.0
    else:
        def_H1, def_H2, def_m = 8.0, 3.0, 3.0
        def_ds_d = "2.5, 2.5, 2.0, 2.0, 3.0, 4.0, 4.0"
        def_ds_n = "17.0, 9.0"
        def_Ld_A, def_Ln_A = 11.0, 9.0
        def_Ld_B, def_Ln_B = 4.0, 0.0
        def_LAB_th, def_LAB_tt, def_ht = 9.33, 8.0, 3.0
        def_d_ra, def_T1, def_T2 = 3.0, 11.0, 12.0

    st.subheader("1️⃣ Thông số cột nước, hệ số quy đổi và các đoạn đường viền thấm")
    c1, c2, c3 = st.columns(3)
    with c1:
        H1 = st.number_input("Cột nước thượng lưu H₁ (m)", value=def_H1, step=0.5)
        H2 = st.number_input("Cột nước hạ lưu H₂ (m)", value=def_H2, step=0.5)
        m_lane = st.number_input("Hệ số m (Bligh: m=1; Lane: m=1÷3.5)", value=def_m, step=0.5, min_value=1.0)
        loai_dat = st.selectbox("Loại đất nền (Tra Bảng 3-1)", list(BANG_HE_SO_C_LANE.keys()), index=6)
    with c2:
        str_d = st.text_input("Danh sách các đoạn thẳng đứng L_đi (cách nhau dấu phẩy)", value=def_ds_d)
        str_n = st.text_input("Danh sách các đoạn nằm ngang L_ni (cách nhau dấu phẩy)", value=def_ds_n)
        try:
            list_d = [float(x.strip()) for x in str_d.split(",") if x.strip()]
            list_n = [float(x.strip()) for x in str_n.split(",") if x.strip()]
        except ValueError:
            st.error("Vui lòng nhập các số cách nhau bằng dấu phẩy!")
            list_d, list_n = [20.0], [26.0]
        Ld_total = sum(list_d)
        Ln_total = sum(list_n)
        st.info(f"**Tổng đoạn đứng:** $\\sum L_đ = {Ld_total:.2f}\\text{{ m}}$ | **Tổng đoạn ngang:** $\\sum L_n = {Ln_total:.2f}\\text{{ m}}$")
    with c3:
        Ld_A = st.number_input("Tổng đoạn đứng từ A về Hạ lưu: L_đ(A) (m)", value=def_Ld_A, step=0.5)
        Ln_A = st.number_input("Tổng đoạn ngang từ A về Hạ lưu: L_n(A) (m)", value=def_Ln_A, step=0.5)
        Ld_B = st.number_input("Tổng đoạn đứng từ B về Hạ lưu: L_đ(B) (m)", value=def_Ld_B, step=0.5)
        Ln_B = st.number_input("Tổng đoạn ngang từ B về Hạ lưu: L_n(B) (m)", value=def_Ln_B, step=0.5)

    st.subheader("2️⃣ Thông số bản đáy, tầng thấm và cửa ra")
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        L_AB_th = st.number_input("Chiều dài đáy tính W_th (m)", value=def_LAB_th, step=0.5)
        L_AB_tt = st.number_input("Chiều dài đáy tính W_tt (m)", value=def_LAB_tt, step=0.5)
    with b2:
        h_t = st.number_input("Chiều sâu ngập bản đáy h_t (m)", value=def_ht, step=0.5)
        gamma = st.number_input("Trọng lượng riêng nước γ (kN/m³)", value=10.0, step=0.1)
    with b3:
        d_ra = st.number_input("Chiều sâu cừ + chân khay cửa ra d (m)", value=def_d_ra, step=0.5)
        J_ra_gh = st.number_input("Gradien cho phép tại cửa ra [J_ra]", value=0.45, step=0.01)
    with b4:
        T1_day = st.number_input("Chiều dày tầng thấm đầu đáy T₁ (m)", value=def_T1, step=0.5)
        T2_day = st.number_input("Chiều dày tầng thấm cuối đáy T₂ (m)", value=def_T2, step=0.5)
        k_val = st.number_input("Hệ số thấm k (m/s)", value=0.0005, format="%.5f")

    # Tính toán Bligh - Lane
    H = H1 - H2
    C_val = BANG_HE_SO_C_LANE[loai_dat]
    L_tt = Ld_total + (Ln_total / m_lane)
    L_yeu_cau = C_val * H

    x_tt_A = Ld_A + (Ln_A / m_lane)
    x_tt_B = Ld_B + (Ln_B / m_lane)

    h_A = (H / L_tt) * x_tt_A if L_tt > 0 else 0.0
    h_B = (H / L_tt) * x_tt_B if L_tt > 0 else 0.0

    W_th = 0.5 * (h_A + h_B) * gamma * L_AB_th
    W_tt = gamma * h_t * L_AB_tt
    W_dn = W_th + W_tt

    b_ngang = Ln_total
    lam = (1.0 + np.sqrt(1.0 + (b_ngang / d_ra) ** 2)) / 2.0 if d_ra > 0 else 1.0
    J_ra = H / (lam * d_ra * np.pi) if d_ra > 0 else 0.0

    T_tb = 0.5 * (T1_day + T2_day)
    J_d = H / L_tt if L_tt > 0 else 0.0
    J_n = J_d / m_lane
    q_tham = k_val * J_n * T_tb

    tab1, tab2 = st.tabs(["📊 Kết Quả & Biểu Đồ Áp Lực", "📝 Xuất Thuyết Minh Chi Tiết (Bligh - Lane)"])

    with tab1:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Chiều dài tính toán L_tt", f"{L_tt:.2f} m", f"Yêu cầu C.H = {L_yeu_cau:.1f} m")
        m2.metric("Cột nước tại A / B", f"{h_A:.3f}m / {h_B:.3f}m")
        m3.metric("Tổng áp lực đẩy ngược W_đn", f"{W_dn:.2f} kN/m", f"W_th={W_th:.2f} | W_tt={W_tt:.2f}")
        m4.metric("Gradien cửa ra J_ra", f"{J_ra:.3f}", "Đạt" if J_ra <= J_ra_gh else "Xói ngầm!")

        col_res, col_chart = st.columns([1.1, 1.3])
        with col_res:
            st.markdown(f"""
            - **Chiều dài đường viền thấm tính toán ($m = {m_lane}$):**  
              $$L_{{tt}} = \\sum L_đ + \\frac{{\\sum L_n}}{{m}} = {Ld_total:.2f} + \\frac{{{Ln_total:.2f}}}{{{m_lane}}} = \\mathbf{{{L_tt:.2f} \\text{{ m}}}}$$
            - **Cột nước thấm tại A ($x_{{ttA}} = {x_tt_A:.2f}\\text{{ m}}$) và B ($x_{{ttB}} = {x_tt_B:.2f}\\text{{ m}}$):**  
              $$h_A = \\frac{{{H:.2f}}}{{{L_tt:.2f}}} \\times {x_tt_A:.2f} = \\mathbf{{{h_A:.3f} \\text{{ m}}}} \\quad ; \\quad h_B = \\frac{{{H:.2f}}}{{{L_tt:.2f}}} \\times {x_tt_B:.2f} = \\mathbf{{{h_B:.3f} \\text{{ m}}}}$$
            - **Tổng áp lực đẩy ngược lên bản đáy AB:**  
              $$W_{{dn}} = W_{{th}} + W_{{tt}} = {W_th:.2f} + {W_tt:.2f} = \\mathbf{{{W_dn:.2f} \\text{{ kN/m}}}}$$
            - **Kiểm tra độ bền thấm chung:** $L_{{tt}} = {L_tt:.2f}\\text{{ m}}$ {'≥' if L_tt >= L_yeu_cau else '<'} $C \\cdot H = {L_yeu_cau:.2f}\\text{{ m}}$ $\\rightarrow$ **{'ĐẢM BẢO' if L_tt >= L_yeu_cau else 'KHÔNG ĐẢM BẢO'}**
            - **Kiểm tra xói ngầm cửa ra (CT 3-27):** $\\lambda = {lam:.2f} \\Rightarrow J_{{ra}} = \\mathbf{{{J_ra:.3f}}}$ {'≤' if J_ra <= J_ra_gh else '>'} $[J_{{ra}}] = {J_ra_gh}$
            - **Lưu lượng thấm đơn vị:** $T_{{tb}} = {T_tb:.2f}\\text{{ m}}, J_n = {J_n:.4f} \\Rightarrow q = \\mathbf{{{q_tham:.6f} \\text{{ m}}^3/(\\text{{s}}\\cdot\\text{{m}})}}$
            """)
        with col_chart:
            st.pyplot(ve_bieu_do_ap_luc_day_nguoc(L_AB_th, h_A, h_B, h_t, gamma))

    with tab2:
        thuyet_minh_lane = f"""# THUYẾT MINH TÍNH TOÁN THẤM DƯỚI ĐÁY CÔNG TRÌNH
**Phương pháp:** Tỷ lệ đường thẳng (Phương pháp Bligh – Lane, $m = {m_lane}$)

---

### 1. Chiều dài tính toán của đường viền thấm ($L_{{tt}}$)
- Tổng chiều dài các đoạn thẳng đứng: $\\sum L_đ = {" + ".join([str(x) for x in list_d])} = {Ld_total:.2f}\\text{{ m}}$
- Tổng chiều dài các đoạn nằm ngang: $\\sum L_n = {" + ".join([str(x) for x in list_n])} = {Ln_total:.2f}\\text{{ m}}$
- Chiều dài đường viền thấm tính toán:
  $$L_{{tt}} = \\sum L_đ + \\frac{{1}}{{m}}\\sum L_n = {Ld_total:.2f} + \\frac{{{Ln_total:.2f}}}{{{m_lane}}} = \\mathbf{{{L_tt:.2f}\\text{{ m}}}}$$

---

### 2. Tính tổng áp lực đẩy ngược tác dụng lên bản đáy (AB)
- **Chiều dài quy đổi từ điểm A và điểm B tới cửa ra hạ lưu:**
  $$x_{{ttA}} = {Ld_A} + \\frac{{{Ln_A}}}{{{m_lane}}} = {x_tt_A:.2f}\\text{{ m}} \\quad ; \\quad x_{{ttB}} = {Ld_B} + \\frac{{{Ln_B}}}{{{m_lane}}} = {x_tt_B:.2f}\\text{{ m}}$$
- **Cột nước thấm tại A và B:**
  $$h_A = \\frac{{H}}{{L_{{tt}}}} x_{{ttA}} = \\frac{{{H:.2f}}}{{{L_tt:.2f}}} \\times {x_tt_A:.2f} = \\mathbf{{{h_A:.3f}\\text{{ m}}}}$$
  $$h_B = \\frac{{H}}{{L_{{tt}}}} x_{{ttB}} = \\frac{{{H:.2f}}}{{{L_tt:.2f}}} \\times {x_tt_B:.2f} = \\mathbf{{{h_B:.3f}\\text{{ m}}}}$$
- **Áp lực thấm dưới bản đáy AB:**
  $$W_{{th}} = \\frac{{h_A + h_B}}{{2}} \\cdot \\gamma \\cdot L_{{AB}} = \\frac{{{h_A:.3f} + {h_B:.3f}}}{{2}} \\times {gamma} \\times {L_AB_th} = \\mathbf{{{W_th:.2f}\\text{{ kN/m}}}}$$
- **Áp lực thủy tĩnh đẩy ngược dưới bản đáy AB:**
  $$W_{{tt}} = \\gamma \\cdot h_t \\cdot L_{{AB}} = {gamma} \\times {h_t} \\times {L_AB_tt} = \\mathbf{{{W_tt:.2f}\\text{{ kN/m}}}}$$
- **Tổng áp lực đẩy ngược:**
  $$W_{{dn}} = W_{{tt}} + W_{{th}} = {W_tt:.2f} + {W_th:.2f} = \\mathbf{{{W_dn:.2f}\\text{{ kN/m}}}}$$

---

### 3. Kiểm tra độ bền thấm của nền
- **Độ bền thấm chung (Công thức 3-21):**
  Tra Bảng 3-1 với *{loai_dat}* $\\rightarrow C = {C_val}$.
  $$L_{{tt}} = {L_tt:.2f}\\text{{ m}} {"\\ge" if L_tt >= L_yeu_cau else "<"} C \\cdot H = {C_val} \\times {H:.2f} = {L_yeu_cau:.2f}\\text{{ m}}$$
  $\\rightarrow$ Đường viền thấm **{"ĐỦ DÀI" if L_tt >= L_yeu_cau else "CHƯA ĐỦ DÀI"}** để đảm bảo độ bền thấm chung.
- **Độ bền thấm tại cửa ra (Công thức 3-27):**
  Với $b = {b_ngang}\\text{{ m}}, d = {d_ra}\\text{{ m}}$:
  $$\\lambda = \\frac{{1 + \\sqrt{{1 + (b/d)^2}}}}{{2}} = \\frac{{1 + \\sqrt{{1 + ({b_ngang}/{d_ra})^2}}}}{{2}} = {lam:.2f}$$
  $$J_{{ra}} = \\frac{{H}}{{\\lambda \\cdot d \\cdot \\pi}} = \\frac{{{H:.2f}}}{{{lam:.2f} \\times {d_ra} \\times 3,1416}} = \\mathbf{{{J_ra:.3f}}} {"\\le" if J_ra <= J_ra_gh else ">"} [J_{{ra}}] = {J_ra_gh}$$
  $\\rightarrow$ Tại cửa ra **{"KHÔNG xảy ra xói ngầm cơ học" if J_ra <= J_ra_gh else "XẢY RA xói ngầm cơ học (Cần kéo dài đường viền thấm và làm tầng lọc ngược tại cửa ra)"}**.

---

### 4. Tính lưu lượng thấm đơn vị dưới đáy cống
- Chiều dày bình quân của tầng thấm: $T_{{tb}} = \\frac{{T_1 + T_2}}{{2}} = \\frac{{{T1_day} + {T2_day}}}{{2}} = {T_tb:.2f}\\text{{ m}}$
- Gradien thấm trung bình qua các đoạn nằm ngang: $J_n = \\frac{{J_đ}}{{m}} = \\frac{{H}}{{m \\cdot L_{{tt}}}} = \\frac{{{H:.2f}}}{{{m_lane} \\times {L_tt:.2f}}} = {J_n:.4f}$
- Lưu lượng thấm đơn vị dưới đáy cống:
  $$q = k \\cdot J_n \\cdot T_{{tb}} = {k_val:.5f} \\times {J_n:.4f} \\times {T_tb:.2f} = \\mathbf{{{q_tham:.6f}\\text{{ m}}^3/(\\text{{s}}\\cdot\\text{{m}})}}$$
"""
        st.markdown(thuyet_minh_lane)
        st.download_button(
            label="📥 Tải Thuyết Minh Tính Toán (.md / Hỗ trợ chuyển sang Word/LaTeX)",
            data=thuyet_minh_lane.encode("utf-8"),
            file_name="Thuyet_minh_tinh_tham_Lane.md",
            mime="text/markdown"
        )
