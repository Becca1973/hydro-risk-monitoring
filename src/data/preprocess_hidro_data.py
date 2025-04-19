import os
import re
import pandas as pd
import numpy as np
from lxml import etree as ET


def preprocess_hidro_data():
    # Odpri XML datoteko
    with open("data/raw/hidro/hidro_data.xml", "rb") as file:
        tree = ET.parse(file)
        root = tree.getroot()

    # Osnovne informacije iz XML
    print(f"Verzija: {root.attrib.get('verzija', 'neznano')}")
    print(f"Vir: {root.findtext('vir', default='neznano')}")
    print(f"Predlagan zajem: {root.findtext('predlagan_zajem', default='')}")
    print(
        f"Perioda zajema: {root.findtext('predlagan_zajem_perioda', default='')}")
    print(f"Datum priprave: {root.findtext('datum_priprave', default='')}")

    output_dir = "data/preprocessed/hidro"
    os.makedirs(output_dir, exist_ok=True)

    for postaja in root.findall("postaja"):
        sifra = postaja.get("sifra", "neznano")
        merilno_mesto_raw = postaja.findtext(
            "merilno_mesto", default="neznano")

        # Očisti ime datoteke (zamenja posebne znake, ne pa šumnikov)
        merilno_mesto = re.sub(r"[^\wšđžčćŠĐŽČĆ]+",
                               "_", merilno_mesto_raw, flags=re.UNICODE)

        # Zberi podatke
        podatki = {
            "sifra": sifra,
            "datum": postaja.findtext("datum", default=""),
            "datum_cet": postaja.findtext("datum_cet", default=""),
            "vodostaj": postaja.findtext("vodostaj", default=""),
            "pretok": postaja.findtext("pretok", default=""),
            "prvi_vv_pretok": postaja.findtext("prvi_vv_pretok", default=""),
            "drugi_vv_pretok": postaja.findtext("drugi_vv_pretok", default=""),
            "tretji_vv_pretok": postaja.findtext("tretji_vv_pretok", default=""),
            "pretok_znacilni": postaja.findtext("pretok_znacilni", default=""),
            "temp_vode": postaja.findtext("temp_vode", default=""),
        }

        df_new = pd.DataFrame([podatki])

        # Določi pot do datoteke
        filename = f"{sifra}_{merilno_mesto}.csv"
        filepath = os.path.join(output_dir, filename)

        # Če obstaja CSV, ga preberi in združi
        if os.path.exists(filepath):
            df_existing = pd.read_csv(filepath, encoding="utf-8")
            df = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df = df_new

        # Odstrani podvojene po datumu
        df = df.drop_duplicates(subset=["datum"])
        df = df.sort_values(by="datum")

        # (neobvezno) Zamenjaj prazne nize z NaN
        df = df.replace("", np.nan)

        # Shrani CSV
        df.to_csv(filepath, index=False, encoding="utf-8")

        print(f"[✓] Shranjeno: {filename} ({len(df)} skupnih vrstic)")

    print("✅ Vsi podatki so bili uspešno posodobljeni in shranjeni.")


if __name__ == "__main__":
    preprocess_hidro_data()
