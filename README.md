# 🌍 Procedural Planet Generator

Um gerador de planetas 3D que roda direto no navegador. Você ajusta uns controles, e um planeta único vai se formando na tela. Dá pra colocar árvores na terra e barquinhos na água também.

**Acesse aqui:** [https://christiankringel.github.io/Procedural-Planet-Generator/](https://christiankringel.github.io/Procedural-Planet-Generator/)

---

## O que é?

Um estúdio visual pra criar e brincar com planetas em 3D. O terreno é gerado automaticamente com montanhas, vales e oceanos, e conforme você mexe nos controles o planeta vai mudando na sua frente.

---

## Controles

### 🌱 Seed (Semente)

Um campo de texto onde você digita qualquer palavra ou frase A mesma seed sempre gera o mesmo planeta. Troque a seed e aparece um planeta completamente diferente.

### 📐 Resolution (Resolução)

Um slider que vai de **20** a **200**. Ele controla o nível de detalhe da superfície. Valores baixos deixam o planeta mais "grosseiro" e angular, valores altos deixam mais suave e bonito (mas pode pesar mais no computador).

### 🔊 Noise (Ruído)

Aqui você mexe na forma dos continentes e do terreno.

**Noise Type (Tipo)** é um dropdown com três estilos de terreno:
  - *Perlin* gera um terreno suave e natural
  - *Simplex* é parecido com o Perlin mas com formas um pouco diferentes
  - *Random* cria um terreno mais caótico e irregular

**Strength (Intensidade)** é um slider de **0** a **2**. Quanto maior, mais exagerados ficam os relevos, com montanhas mais altas e vales mais profundos. Em zero, o planeta fica uma esfera lisa.

### 🌊 Water vs Land (Água vs Terra)

Um slider de **-1** a **1** que define onde começa o oceano. Puxa pra direita e mais água aparece (o oceano "sobe"). Puxa pra esquerda e mais terra fica exposta.

### 🌳 Objects (Objetos)

Um slider de **0** a **2000** que define quantos objetos são espalhados pelo planeta automaticamente. Árvores caem na terra e barcos ficam na água.

### ⛰️ Relief Aggressiveness (Agressividade do Relevo)

Um slider de **0** a **2** que controla o quão pronunciados são picos e montanhas. Valores maiores deixam o terreno mais dramático e pontiagudo.

### 🚀 Missile Duration (Duração do Míssil)

Um slider de **0.3s** a **5.0s** que define a duração/velocidade dos mísseis lançados com a tecla **M**. Menor valor = míssil mais rápido.

### 🌀 Rotation Speed (Velocidade de Rotação)

Um slider de **0×** a **3×** que controla a velocidade da rotação automática do planeta quando a rotação automática está ativada.

### ☀️ Shadows (Sombras)

Liga ou desliga as sombras do sol no planeta. Com sombras ligadas o visual fica bem mais bonito.

### 🔄 Auto Rotation (Rotação Automática)

Liga ou desliga a rotação automática do planeta. Desativa se quiser deixar ele paradinho enquanto explora.

---

## Botões de Ação

| Botão | O que faz |
|-------|-----------|
| **Regenerate** | Reconstrói o planeta do zero com as configurações atuais |
| **Random Seed** | Gera uma seed aleatória, criando um planeta novo na hora |
| **Reset Preset** | Volta tudo pros valores padrão |

---

## Mouse

| Ação | O que faz |
|------|-----------|
| **Arrastar (botão esquerdo)** | Gira a câmera ao redor do planeta |
| **Scroll** | Zoom, aproxima ou afasta |
| **Clique direito** | Coloca uma árvore (na terra) ou um barco (na água) onde você clicou |

| **Tecla M** | Lança um míssil na direção do cursor; cria crateras ao atingir a superfície |
| **Tecla P** | Pausa/retoma a rotação automática do planeta |

---

## Informações na Tela

No rodapé da barra lateral tem uns indicadores:

- **FPS** mostra quantos quadros por segundo estão sendo renderizados
- **Seed** mostra a seed atual do planeta
- **Triangles** mostra quantos triângulos formam a malha do planeta
- **Objects** mostra quantos objetos (árvores e barcos) existem no planeta

---

## Como rodar (desenvolvimento)

Siga estes passos para executar o projeto localmente:

1. Instale dependências:

```bash
npm install
```

2. Rode em modo desenvolvimento (servidor + cliente):

```bash
npm run dev
```

3. Abra o navegador em:

```
http://localhost:5000
```

Observações:

- O script `dev` inicia o servidor que integra o cliente (Vite) e a API. A porta padrão é `5000`.
- Para preparar uma versão de produção, use `npm run build` e depois `npm start`.
