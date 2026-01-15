import general

# NON-TERMINALS
variable = [
    "K", "P", "S", "Pel", "Ket", "PP", 
    "NP_S", "NP_Pel", "NP_Loc", "NP_Time", "NumP",
    "Prep", "Noun", "PropNoun", "Pronoun", "Adj", "Det", "Num", "Adv", "NounTime",
    "X1", "X2", "X3", "X4"
]

# CNF
production = {
    "K": [
        "PS",           # K -> P S
        "X1Pel",        # K -> X1 Pel
        "X2S",          # K -> X2 S
        "X1X3",         # K -> X1 X3
        "X2X4"          # K -> X2 X4
    ],
    
    "X1": [
        "PS"            # X1 -> P S
    ],
    
    "X2": [
        "PPel"          # X2 -> P Pel
    ],
    
    "X3": [
        "PelKet"        # X3 -> Pel Ket
    ],
    
    "X4": [
        "SKet"          # X4 -> S Ket
    ],
    
    "P": [
        "PrepNP_Loc"    # P -> Prep NP_Loc
    ],
    
    "PP": [
        "PrepNP_Loc"    # PP -> Prep NP_Loc
    ],
    
    "S": (
        general.kata_benda +      # S -> 'kata_benda'
        general.proper_noun +     # S -> 'proper_noun'
        general.kata_ganti +      # S -> 'kata_ganti'
        [
            "DetNP_S",            # S -> Det NP_S
            "NP_SDet",            # S -> NP_S Det
            "NP_SAdj",            # S -> NP_S Adj
            "NP_SNoun",           # S -> NP_S Noun
            "NP_SPropNoun",       # S -> NP_S PropNoun
            "NP_SPronoun"         # S -> NP_S Pronoun
        ]
    ),
    
    "NP_S": (
        general.kata_benda +      # NP_S -> 'kata_benda'
        general.proper_noun +     # NP_S -> 'proper_noun'
        general.kata_ganti +      # NP_S -> 'kata_ganti'
        [
            "DetNP_S",            # NP_S -> Det NP_S
            "NP_SDet",            # NP_S -> NP_S Det
            "NP_SAdj",            # NP_S -> NP_S Adj
            "NP_SNoun",           # NP_S -> NP_S Noun
            "NP_SPropNoun",       # NP_S -> NP_S PropNoun
            "NP_SPronoun"         # NP_S -> NP_S Pronoun
        ]
    ),
    
    "Pel": (
        general.kata_benda +      # Pel -> 'kata_benda'
        general.numeralia +       # Pel -> 'numeralia'
        [
            "NP_PelAdj",          # Pel -> NP_Pel Adj
            "NP_PelNoun",         # Pel -> NP_Pel Noun
            "NumPNum",            # Pel -> NumP Num
            "NumPNoun"            # Pel -> NumP Noun
        ]
    ),
    
    "NP_Pel": (
        general.kata_benda +      # NP_Pel -> 'kata_benda'
        [
            "NP_PelAdj",          # NP_Pel -> NP_Pel Adj
            "NP_PelNoun"          # NP_Pel -> NP_Pel Noun
        ]
    ),
    
    "NumP": (
        general.numeralia +       # NumP -> 'numeralia'
        [
            "NumPNum",            # NumP -> NumP Num
            "NumPNoun"            # NumP -> NumP Noun
        ]
    ),
    
    "NP_Time": (
        general.kata_benda_waktu +    # NP_Time -> 'kata_benda_waktu'
        [
            "NP_TimeNounTime"         # NP_Time -> NP_Time NounTime
        ]
    ),
    
    "Ket": (
        general.kata_benda_waktu +    # Ket -> 'kata_benda_waktu'
        [
            "NP_TimeNounTime"         # Ket -> NP_Time NounTime
        ]
    ),
    
    "NP_Loc": (
        general.kata_benda +      # NP_Loc -> 'kata_benda'
        general.proper_noun +     # NP_Loc -> 'proper_noun'
        [
            "NP_LocAdj",          # NP_Loc -> NP_Loc Adj
            "NP_LocDet"           # NP_Loc -> NP_Loc Det
        ]
    ),
    
    "Noun": general.kata_benda,              # Noun -> 'kata_benda'
    "PropNoun": general.proper_noun,         # PropNoun -> 'proper_noun'
    "Pronoun": general.kata_ganti,           # Pronoun -> 'kata_ganti'
    "Adj": general.kata_sifat,               # Adj -> 'kata_sifat'
    "Num": general.numeralia,                # Num -> 'numeralia'
    "Prep": general.kata_preposisi,          # Prep -> 'kata_preposisi'
    "Adv": general.adverbia,                 # Adv -> 'adverbia'
    "Det": general.determinan,               # Det -> 'determiner'
    "NounTime": general.kata_benda_waktu     # NounTime -> 'kata_benda_waktu'
}

start_symbol = ["K"]

def check_production(array):
    """
    Mengecek apakah elemen dalam array (gabungan string) ada di value production.
    Contoh: array=["PS"], akan mengecek siapa yang punya "PS". Jawabannya "K".
    
    Args:
        array (list): List string yang akan dicek dalam produksi
        
    Returns:
        list: List non-terminal yang bisa memproduksi string dalam array
    """
    sum_result = set()
    for i in array:
        for j in variable:
            prod_list = production.get(j)
            if prod_list and i in prod_list:
                sum_result.add(j)
    return list(sum_result)

def check_symbol(array):
    """
    Mengecek apakah simbol hasil parsing terakhir adalah Start Symbol (K).
    
    Args:
        array (list): List simbol hasil parsing
        
    Returns:
        bool: True jika mengandung start symbol, False jika tidak
    """
    for i in array:
        if i in start_symbol:
            return True
    return False