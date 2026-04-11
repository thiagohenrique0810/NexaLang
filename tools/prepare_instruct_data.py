#!/usr/bin/env python3
"""
Prepare instruction-tuning data for NexaLang Transformer v5.

Extracts dialogue pairs from Portuguese novels and adds synthetic
conversational data. Format: [P] question\n[R] answer\n\n

Output: byte-level binary files (V=256) for fine-tuning.
"""
import struct
import os
import random
import re

CORPUS_PATH = "data/text_pt_v2/corpus_pt_v2.txt"
OUTPUT_DIR = "data/instruct_pt"
VOCAB_SIZE = 256
TRAIN_RATIO = 0.9

# ─── Synthetic conversational pairs ───────────────────────────────────
SYNTHETIC_PAIRS = [
    # Saudações
    ("Olá!", "Olá! Como posso ajudar?"),
    ("Olá, como vai?", "Vou bem, obrigado! E você, como está?"),
    ("Oi!", "Oi! Tudo bem?"),
    ("Oi, tudo bem?", "Tudo bem sim! O que deseja saber?"),
    ("Bom dia!", "Bom dia! Espero que tenha um ótimo dia."),
    ("Bom dia, como está?", "Estou bem, obrigado! Bom dia para você também."),
    ("Boa tarde!", "Boa tarde! Em que posso ajudar?"),
    ("Boa noite!", "Boa noite! Posso ajudar com alguma coisa?"),
    ("E aí?", "E aí! Tudo certo? O que posso fazer por você?"),
    ("Como vai você?", "Vou muito bem, obrigado por perguntar!"),
    ("Tudo bem?", "Tudo ótimo! E com você?"),
    ("Tudo certo?", "Tudo certo sim! Como posso ajudar?"),

    # Identidade
    ("Quem é você?", "Sou um modelo de linguagem treinado com literatura portuguesa."),
    ("Qual é o seu nome?", "Me chamam de NexaLang. Sou um modelo de linguagem."),
    ("O que você é?", "Sou um modelo de inteligência artificial treinado para conversar em português."),
    ("Você é humano?", "Não, sou um modelo de linguagem artificial. Fui treinado com textos em português."),
    ("Você é uma inteligência artificial?", "Sim, sou uma inteligência artificial treinada com literatura brasileira e portuguesa."),
    ("Como você funciona?", "Funciono prevendo a próxima palavra com base no contexto da conversa. Fui treinado com muitos textos em português."),
    ("Quem te criou?", "Fui criado usando a linguagem NexaLang, uma linguagem de programação de sistemas."),
    ("Em que idioma você fala?", "Falo português! Fui treinado principalmente com literatura brasileira e portuguesa."),

    # Capacidades
    ("O que você sabe fazer?", "Posso conversar em português, responder perguntas e discutir diversos assuntos."),
    ("Você pode me ajudar?", "Claro! Diga-me o que precisa e farei o meu melhor para ajudar."),
    ("Pode me contar uma história?", "Era uma vez um jovem que vivia numa pequena cidade do interior. Um dia, decidiu partir em busca de aventuras e conhecimento. Viajou por muitas terras, conheceu pessoas incríveis e aprendeu lições valiosas sobre a vida."),
    ("Me conte algo interessante.", "Sabia que Machado de Assis é considerado um dos maiores escritores da língua portuguesa? Ele escreveu obras como Dom Casmurro e Memórias Póstumas de Brás Cubas, que são lidas até hoje em todo o mundo."),

    # Perguntas gerais
    ("O que é literatura?", "Literatura é a arte de escrever e criar textos que expressam ideias, emoções e histórias. Inclui romances, poesia, contos e outros gêneros."),
    ("Quem foi Machado de Assis?", "Machado de Assis foi um dos maiores escritores brasileiros. Nasceu no Rio de Janeiro e escreveu obras como Dom Casmurro, Memórias Póstumas de Brás Cubas e Quincas Borba."),
    ("Quem foi Eça de Queirós?", "Eça de Queirós foi um dos mais importantes escritores portugueses do século XIX. Escreveu obras como Os Maias, O Primo Basílio e O Crime do Padre Amaro."),
    ("O que é um romance?", "Um romance é uma obra literária longa que conta uma história com personagens, conflitos e desenvolvimento de enredo."),
    ("O que é poesia?", "Poesia é uma forma de expressão literária que usa a linguagem de modo artístico, muitas vezes com ritmo, rima e figuras de linguagem."),
    ("O que é o Brasil?", "O Brasil é o maior país da América do Sul e o quinto maior do mundo. É conhecido por sua rica cultura, diversidade e literatura."),
    ("O que é Portugal?", "Portugal é um país europeu na Península Ibérica. É o berço da língua portuguesa e tem uma rica tradição literária."),
    ("Qual é a capital do Brasil?", "A capital do Brasil é Brasília, fundada em 1960. Antes disso, a capital era o Rio de Janeiro."),
    ("Qual é a capital de Portugal?", "A capital de Portugal é Lisboa, uma cidade histórica às margens do rio Tejo."),

    # Opiniões e reflexões
    ("Qual é o sentido da vida?", "Essa é uma das grandes perguntas da filosofia. Cada pessoa encontra seu próprio sentido através de suas experiências, relações e busca por conhecimento."),
    ("O que é felicidade?", "A felicidade é um estado de satisfação e bem-estar. Os filósofos dizem que ela pode vir do conhecimento, das relações humanas e de viver com propósito."),
    ("O que é amor?", "O amor é um dos sentimentos mais profundos do ser humano. Pode ser amor por outra pessoa, pela família, pela arte ou pela vida."),
    ("Você gosta de ler?", "Como modelo de linguagem, fui treinado com muitos textos. Posso dizer que a leitura é fundamental para o conhecimento e a compreensão do mundo."),
    ("Qual seu livro favorito?", "Fui treinado com muitas obras maravilhosas. Dom Casmurro de Machado de Assis é uma obra fascinante pela forma como explora a natureza humana."),

    # Conversação casual
    ("Como está o tempo hoje?", "Não tenho acesso a informações sobre o tempo atual, mas espero que esteja um dia agradável!"),
    ("Que horas são?", "Não tenho acesso ao relógio, mas posso ajudar com outras perguntas."),
    ("Estou triste.", "Lamento ouvir isso. Lembre-se que momentos difíceis são passageiros. Posso tentar ajudar a conversar sobre o que sente."),
    ("Estou feliz!", "Que bom! Fico contente em saber. A felicidade é contagiante!"),
    ("Estou entediado.", "Que tal ler um bom livro? Posso sugerir algumas obras da literatura portuguesa que são muito interessantes."),
    ("Obrigado!", "De nada! Fico feliz em ajudar."),
    ("Obrigado pela ajuda.", "Foi um prazer ajudar! Se precisar de mais alguma coisa, estou aqui."),
    ("Muito obrigado!", "De nada! Sempre que precisar, pode contar comigo."),
    ("Tchau!", "Tchau! Foi bom conversar com você. Até a próxima!"),
    ("Até mais!", "Até mais! Tenha um ótimo dia!"),
    ("Até logo!", "Até logo! Volte sempre que quiser conversar."),

    # Pedidos
    ("Me diga algo em português.", "A língua portuguesa é uma das mais belas do mundo. Com mais de 260 milhões de falantes, é a língua oficial de nove países."),
    ("Fale sobre a natureza.", "A natureza é fonte de vida e inspiração. As florestas, os rios, as montanhas e os oceanos formam um ecossistema complexo e belo que devemos preservar."),
    ("Fale sobre música.", "A música é uma forma universal de expressão artística. No Brasil, temos gêneros ricos como o samba, a bossa nova, o forró e muitos outros."),
    ("O que pensa sobre a tecnologia?", "A tecnologia é uma ferramenta poderosa que pode melhorar a vida das pessoas quando usada com sabedoria e responsabilidade."),

    # Perguntas sobre a conversa
    ("Pode repetir?", "Claro! O que gostaria que eu repetisse?"),
    ("Não entendi.", "Peço desculpas. Posso tentar explicar de outra forma. O que não ficou claro?"),
    ("Explique melhor.", "Com prazer! Sobre qual assunto gostaria que eu me aprofundasse?"),
    ("Me conte mais.", "Claro! Sobre qual tema gostaria de saber mais?"),
    ("O que mais você pode me dizer?", "Posso falar sobre muitos assuntos. O que lhe interessa? Literatura, história, filosofia, ou outro tema?"),

    # Perguntas com respostas mais elaboradas
    ("Como aprender português?", "Para aprender português, recomendo ler muitos textos, praticar a escrita e a conversação, e estudar a gramática aos poucos. A literatura brasileira e portuguesa é uma ótima fonte de aprendizado."),
    ("O que devo ler?", "Recomendo começar com os clássicos: Dom Casmurro de Machado de Assis, O Primo Basílio de Eça de Queirós, e Grande Sertão Veredas de Guimarães Rosa."),
    ("Por que a leitura é importante?", "A leitura amplia o vocabulário, desenvolve o pensamento crítico e a imaginação. Através dos livros, podemos viver experiências e conhecer mundos diferentes."),
    ("O que é inteligência artificial?", "Inteligência artificial é um campo da ciência da computação que busca criar sistemas capazes de realizar tarefas que normalmente requerem inteligência humana, como entender linguagem e aprender."),
    ("Como funciona um modelo de linguagem?", "Um modelo de linguagem aprende padrões estatísticos dos textos durante o treinamento. Quando recebe uma entrada, usa esses padrões para prever as palavras mais prováveis que devem seguir."),
]

# Additional variations by augmenting existing pairs
AUGMENTATION_TEMPLATES = [
    # Question variations
    ("Oi, tudo bom?", "Tudo bom sim! Posso ajudar em algo?"),
    ("Como você está?", "Estou bem, obrigado! Pronto para conversar."),
    ("Olá, boa tarde!", "Boa tarde! Como posso ser útil?"),
    ("Oi, bom dia!", "Bom dia! Espero que esteja bem."),
    ("Ei, pode me ajudar?", "Claro! Estou aqui para ajudar. O que precisa?"),
    ("Preciso de ajuda.", "Estou aqui para ajudar! Diga-me o que precisa."),
    ("Tem alguém aí?", "Estou aqui! Como posso ajudar?"),
    ("Oi, quero conversar.", "Ótimo! Adoro conversar. Sobre o que gostaria de falar?"),
    ("Vamos conversar?", "Vamos sim! Qual assunto lhe interessa?"),
    ("Pode conversar comigo?", "Com certeza! Estou aqui para isso. O que gostaria de discutir?"),

    # Follow-ups
    ("Interessante!", "Que bom que achou interessante! Quer saber mais sobre alguma coisa?"),
    ("Legal!", "Fico feliz que tenha gostado! Posso ajudar com mais alguma coisa?"),
    ("Verdade.", "Sim, é verdade. Há algo mais que gostaria de saber?"),
    ("Concordo.", "Que bom que concordamos! O que mais gostaria de discutir?"),
    ("Não concordo.", "Respeito sua opinião. Pode me dizer por que pensa diferente?"),
    ("Isso é verdade?", "Procuro ser o mais preciso possível, mas posso cometer erros. O que gostaria de verificar?"),
    ("Pode explicar de novo?", "Claro! Vou tentar explicar de uma forma mais clara. O que não ficou bem entendido?"),
    ("Faz sentido.", "Fico feliz que tenha feito sentido! Se tiver mais dúvidas, é só perguntar."),

    # More topics
    ("O que é filosofia?", "Filosofia é o estudo das questões fundamentais sobre a existência, o conhecimento, a verdade, a ética e a mente. Vem do grego e significa amor pela sabedoria."),
    ("O que é ciência?", "Ciência é o estudo sistemático do mundo natural através da observação, experimentação e formulação de teorias para explicar os fenômenos da natureza."),
    ("O que é história?", "História é o estudo dos acontecimentos passados da humanidade. Através dela, compreendemos como as sociedades evoluíram e como chegamos onde estamos."),
    ("O que é arte?", "Arte é a expressão criativa do ser humano através de diversas formas: pintura, escultura, música, literatura, dança e muitas outras manifestações."),
    ("O que é democracia?", "Democracia é um sistema de governo em que o poder pertence ao povo, que o exerce diretamente ou por meio de representantes eleitos."),
    ("O que é educação?", "Educação é o processo de aprendizagem e desenvolvimento de habilidades, conhecimentos e valores. É fundamental para o progresso individual e social."),
    ("Fale sobre o Rio de Janeiro.", "O Rio de Janeiro é uma das cidades mais famosas do Brasil. Conhecida pelo Cristo Redentor, pelo Pão de Açúcar e pelas belas praias como Copacabana e Ipanema."),
    ("Fale sobre Lisboa.", "Lisboa é a capital de Portugal, uma cidade cheia de história, cultura e beleza. É conhecida pelos seus bairros históricos, o Mosteiro dos Jerónimos e os famosos pastéis de Belém."),
]


def extract_dialogue_pairs(text):
    """Extract Q&A pairs from literary dialogue (lines starting with --)."""
    lines = text.split('\n')
    pairs = []

    # Find dialogue lines with their positions
    dialogue_indices = []
    for i, line in enumerate(lines):
        if line.startswith('--'):
            dialogue_indices.append(i)

    # Pair consecutive dialogue lines (gap of 1-3 lines between them)
    used = set()
    for idx in range(len(dialogue_indices) - 1):
        i = dialogue_indices[idx]
        j = dialogue_indices[idx + 1]
        gap = j - i - 1

        if gap <= 3 and i not in used:
            q_line = lines[i]
            a_line = lines[j]

            # Strip -- prefix
            q = q_line.lstrip('-').strip()
            a = a_line.lstrip('-').strip()

            # Skip very short or very long
            if len(q) < 3 or len(a) < 3:
                continue
            if len(q) > 400 or len(a) > 400:
                continue

            # Also collect multi-line dialogue (continuation)
            # Check if next lines (not starting with --) continue the dialogue
            for k in range(i + 1, j):
                cont = lines[k].strip()
                if cont and not cont.startswith('--'):
                    q += ' ' + cont

            pairs.append((q, a))
            used.add(i)
            used.add(j)

    return pairs


def format_pair(q, a):
    """Format a Q&A pair with markers."""
    return f"[P] {q}\n[R] {a}\n\n"


def encode_to_bytes(text):
    """Encode text to byte-level tokens (V=256)."""
    return list(text.encode('utf-8', errors='replace'))


def write_binary(tokens, path):
    """Write tokens in NexaLang binary format."""
    n = len(tokens)
    with open(path, 'wb') as f:
        f.write(struct.pack('<i', n))
        f.write(struct.pack('<i', VOCAB_SIZE))
        f.write(bytes(tokens))
        # Vocab table (identity for byte-level)
        for i in range(VOCAB_SIZE):
            f.write(struct.pack('B', i))
    print(f"  Wrote {path}: {n} tokens ({n / 1024:.1f} KB)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  NexaLang Instruction Data Preparation")
    print("=" * 60)

    # 1. Load corpus and extract dialogue pairs
    print(f"\nLoading corpus from {CORPUS_PATH}...")
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"  Corpus size: {len(text)} chars")

    print("\nExtracting dialogue pairs...")
    dialogue_pairs = extract_dialogue_pairs(text)
    print(f"  Found {len(dialogue_pairs)} dialogue pairs")

    # 2. Combine all pairs
    all_pairs = []

    # Dialogue from novels
    all_pairs.extend(dialogue_pairs)

    # Synthetic conversational pairs (repeat more to increase weight)
    synthetic_all = SYNTHETIC_PAIRS + AUGMENTATION_TEMPLATES
    for _ in range(80):  # Repeat 80x to give them dominant weight
        all_pairs.extend(synthetic_all)

    print(f"  Synthetic pairs: {len(synthetic_all)} × 80 = {len(synthetic_all) * 80}")
    print(f"  Total pairs: {len(all_pairs)}")

    # 3. Shuffle
    random.seed(42)
    random.shuffle(all_pairs)

    # 4. Format and encode
    print("\nFormatting and encoding...")
    full_text = ""
    for q, a in all_pairs:
        full_text += format_pair(q, a)

    tokens = encode_to_bytes(full_text)
    print(f"  Total tokens: {len(tokens)}")
    print(f"  Total text size: {len(full_text)} chars")

    # 5. Split train/val
    split_idx = int(len(tokens) * TRAIN_RATIO)
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

    # 6. Write binary files
    print("\nWriting binary files...")
    write_binary(train_tokens, os.path.join(OUTPUT_DIR, "train.bin"))
    write_binary(val_tokens, os.path.join(OUTPUT_DIR, "val.bin"))

    # 7. Save raw text for inspection
    text_path = os.path.join(OUTPUT_DIR, "instruct_data.txt")
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(full_text)
    print(f"  Wrote {text_path}: {len(full_text)} chars")

    # 8. Sample some pairs
    print("\n" + "=" * 60)
    print("  Sample pairs:")
    print("=" * 60)
    random.seed(0)
    samples = random.sample(all_pairs[:len(dialogue_pairs)], min(5, len(dialogue_pairs)))
    for q, a in samples:
        print(f"\n  [P] {q[:80]}")
        print(f"  [R] {a[:80]}")

    print("\n  --- Synthetic ---")
    for q, a in synthetic_all[:5]:
        print(f"\n  [P] {q}")
        print(f"  [R] {a[:80]}")

    print("\n" + "=" * 60)
    print(f"  Done! Files in {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
