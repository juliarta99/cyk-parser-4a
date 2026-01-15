import streamlit as st
import pandas as pd
import general
import cyk_process

def analyze_sentence_components(node, words):
    """
    Menganalisis komponen kalimat dari parse tree.
    
    Args:
        node (dict): Parse tree node
        words (list): List kata
        
    Returns:
        dict: Dictionary komponen kalimat
    """
    components = {
        "Frasa Preposisional (P/PP)": [],
        "Subjek (S)": [],
        "Pelengkap (Pel)": [],
        "Keterangan (Ket)": []
    }
    
    def traverse(n, current_label=None):
        if n is None:
            return
        
        label = n['label']
        
        if label in ['P', 'PP']:
            current_label = "Frasa Preposisional (P/PP)"
        elif label == 'S' and current_label != "Frasa Preposisional (P/PP)":
            current_label = "Subjek (S)"
        elif label == 'Pel':
            current_label = "Pelengkap (Pel)"
        elif label == 'Ket':
            current_label = "Keterangan (Ket)"
        
        if n['type'] == 'terminal' and current_label:
            if n['word'] not in components[current_label]:
                components[current_label].append(n['word'])
        
        if 'left' in n:
            traverse(n['left'], current_label)
        if 'right' in n:
            traverse(n['right'], current_label)
    
    traverse(node)
    return components

st.set_page_config(
    page_title="SEKEN App - CYK Prepositional Phrase", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("SEKEN App - Aplikasi Validasi Kalimat Bahasa Bali Berpedikat Frase Preposisi")
st.markdown("""
Implementasi **Algoritma Cocke-Younger-Kasami (CYK)** untuk menganalisis struktur kalimat Bahasa Bali dalam bentuk **Chomsky Normal Form (CNF)**.
""")

st.markdown("---")

col1, col2 = st.columns([3, 1])

with col1:
    input_sentence = st.text_input(
        "Masukkan Kalimat Bahasa Bali:", 
        placeholder="Contoh: ring sekolah murid ento"
    )

with col2:
    st.write("")
    st.write("")
    check_button = st.button("Analisis Kalimat", type="primary", use_container_width=True)

if check_button:
    if not input_sentence.strip():
        st.warning("Mohon masukkan kalimat terlebih dahulu.")
    else:
        words = input_sentence.lower().split()
        
        is_known, unknown_words = general.check_alphabet(words)
        
        if not is_known:
            st.error(f"**Kata tidak dikenali dalam kamus:** {', '.join(unknown_words)}")
            st.info("Silakan tambahkan kata tersebut ke folder `file_kata` jika diperlukan.")
        else:
            st.success("Semua kata dikenali dalam kamus!")
            st.write("---")
            
            st.subheader("Tokenisasi")
            st.code(" → ".join(words), language="text")
            
            with st.spinner("Memproses algoritma CYK..."):
                final_table, backpointer = cyk_process.cyk_parse(words)
                n = len(words)

            st.subheader("Triangular Table (CYK Chart)")
            st.caption("Tabel bottom-up: Baris bawah untuk kata tunggal, naik ke atas untuk substring yang lebih panjang.")
            
            display_rows = []
            row_labels = []
            
            for row in range(n-1, -1, -1):
                display_row = []
                length = row + 1
                num_cells = n - row
                
                for col in range(num_cells):
                    cell_content = final_table[row][col]
                    if cell_content:
                        display_row.append(", ".join(sorted(list(cell_content))))
                    else:
                        display_row.append("-")
                
                while len(display_row) < n:
                    display_row.append("")
                
                display_rows.append(display_row)
                row_labels.append(f"Length {length}")
            
            col_labels = [words[i] for i in range(n)]
            
            df_table = pd.DataFrame(display_rows, columns=col_labels, index=row_labels)
            st.dataframe(df_table, use_container_width=True, height=min(400, (n + 1) * 45))
            
            st.caption("**Keterangan:** Length 1 = kata tunggal, Length n = seluruh kalimat")

            st.subheader("Hasil Analisis")
            
            parse_result = cyk_process.get_parse_result(final_table, n)
            is_valid = cyk_process.is_valid_sentence((final_table, backpointer), n, start_symbol="K")
            
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                if is_valid:
                    st.success("**VALID**")
                    st.balloons()
                else:
                    st.error("**TIDAK VALID**")
            
            with col_res2:
                if is_valid:
                    st.write("Kalimat **diterima** oleh grammar. Struktur kalimat sesuai dengan pola yang ditentukan.")
                else:
                    st.write("Kalimat **ditolak** oleh grammar. Struktur tidak sesuai pola yang ditentukan.")
                    st.caption(f"Isi sel terakhir (row {n-1}, col 0): `{parse_result}` → Tidak mengandung start symbol 'K'")

            if is_valid:
                st.write("---")
                st.subheader("Pola Kalimat")
                
                pattern_info = cyk_process.get_sentence_pattern(backpointer, words, start_symbol="K")
                
                if pattern_info:
                    with st.expander("Parse Tree (Pohon Penurunan)", expanded=True):
                        parse_tree_str = cyk_process.format_parse_tree(
                            pattern_info['parse_tree'], 
                            words, 
                            prefix=""
                        )
                        st.code(parse_tree_str, language="text")
                    
                    col_p1, col_p2 = st.columns(2)
                    
                    with col_p1:
                        st.metric(
                            label="Pola Kalimat",
                            value=pattern_info['pattern']
                        )
                    
                    with col_p2:
                        pattern = pattern_info['pattern']
                        if 'P' in pattern and 'S' in pattern:
                            interpretation = "Kalimat Preposisional + Subjek"
                        elif 'S' in pattern and 'Pel' in pattern:
                            interpretation = "Kalimat Subjek + Pelengkap"
                        else:
                            interpretation = "Pola Kompleks"
                        
                        st.metric(
                            label="Interpretasi",
                            value=interpretation
                        )
                    
                    with st.expander("Langkah-Langkah Derivasi"):
                        derivation_steps = pattern_info['derivation']
                        if derivation_steps:
                            for idx, step in enumerate(derivation_steps, 1):
                                indent = "  " * step['depth']
                                st.text(f"{idx}. {indent}{step['rule']}")
                        else:
                            st.info("Tidak ada langkah derivasi (kalimat sangat sederhana)")
                    
                    with st.expander("Analisis Komponen Kalimat"):
                        st.markdown("### Komponen yang Teridentifikasi:")
                        
                        components = analyze_sentence_components(pattern_info['parse_tree'], words)
                        
                        for comp_type, comp_words in components.items():
                            if comp_words:
                                st.markdown(f"**{comp_type}:** {' '.join(comp_words)}")

            with st.expander("Detail Non-Terminal di Sel Akhir"):
                if parse_result:
                    st.write("Non-terminal yang ditemukan:")
                    for nt in sorted(list(parse_result)):
                        st.code(nt, language="text")
                else:
                    st.write("Tidak ada non-terminal yang dapat menurunkan kalimat lengkap.")

with st.sidebar:
    st.header("Tentang Aplikasi")
    
    st.markdown("""
    ### Algoritma CYK
    Cocke-Younger-Kasami adalah algoritma parsing 
    berbasis **dynamic programming** untuk mengecek 
    apakah sebuah string dapat diturunkan oleh 
    Context-Free Grammar dalam bentuk CNF.
    
    ### Struktur Grammar
    - **Start Symbol:** K (Kalimat)
    - **Non-Terminals:** P, S, Pel, Ket, PP, NP, dll.
    - **Terminals:** Kata-kata Bahasa Bali
    
    ### Cara Kerja
    1. Input kalimat diubah menjadi token
    2. Cek validitas kata di kamus
    3. Jalankan algoritma CYK
    4. Analisis struktur gramatikal
    """)
    
    st.markdown("---")
    st.caption("Teori Bahasa dan Otomata")

st.markdown("---")
st.caption("© 2026 - Kelompok 4A | Developed with Streamlit")