df <- read.csv("F1_50_df.csv")
# Função para converter HHMMSS em segundos
time_to_seconds <- function(time) {
  time <- as.integer(time)
  hours <- time %/% 10000
  minutes <- (time %% 10000) %/% 100
  seconds <- time %% 100
  total_seconds <- hours * 3600 + minutes * 60 + seconds
  return(total_seconds)
}
# Adicionar coluna de tempo em segundos
df <- df %>%
  mutate(time_seconds = time_to_seconds(time))
# Ordenar pelo tempo
df <- df %>% arrange(time_seconds)
# Inicializar lista de blocos válidos
blocos_validos <- list()
# Definir tempo inicial e final
start_time <- min(df$time_seconds)
end_time <- max(df$time_seconds)
current_start <- start_time
# Processar blocos de 50 segundos
while (current_start <= end_time) {
  current_end <- current_start + 5
  bloco <- df %>%
    filter(time_seconds >= current_start, time_seconds < current_end)
  # Verifica se há valores > 25 em kneeLangle OU kneeRangle
  if (any(bloco$kneeLangle > 50) || any(bloco$kneeRangle > 50)) {
    blocos_validos[[length(blocos_validos) + 1]] <- bloco
  }
  current_start <- current_start + 5
}
# Juntar e salvar os blocos válidos
if (length(blocos_validos) > 0) {
  resultado <- bind_rows(blocos_validos) %>%
    select(-time_seconds)
  write_csv(resultado, "blocos_filtradosF342.csv")
  message("Arquivo 'blocos_filtradosF342.csv' gerado com sucesso.")
} else {
  message("Nenhum bloco com valores > 25 em kneeLangle ou kneeRangle foi encontrado.")
}