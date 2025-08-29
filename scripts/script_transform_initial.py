# Caminho do arquivo de entrada
input_file = "Pares de frase em Inglêstraduzidas para o Português- - 2025-08-27.tsv"  # ou .csv dependendo do seu arquivo
# Caminho do arquivo de saída
output_file = "pares_organizados.txt"

with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    for line in f_in:
        # Remove espaços e quebras de linha
        line = line.strip()
        if not line:
            continue
        # Divide a linha pelas tabulações
        parts = line.split("\t")
        if len(parts) < 4:
            continue  # Ignora linhas que não tenham todas as colunas
        english = parts[1]
        portuguese = parts[3]
        # Escreve no formato desejado
        f_out.write(f"{english} -> {portuguese}\n")

print("Arquivo convertido com sucesso!")
