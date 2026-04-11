#!/usr/bin/env python3
"""
Prepare instruction-tuning data v2 for NexaLang Transformer v6.

Expanded: ~400 unique synthetic conversational pairs + auto ASCII variants
+ literary dialogue pairs (capped at 2000).
Outputs raw text file for BPE encoding.
"""
import os
import random
import re

CORPUS_PATH = "data/text_pt_v2/corpus_pt_v2.txt"
OUTPUT_DIR = "data/instruct_pt_v2"
SYNTHETIC_REPEAT = 30
MAX_LITERARY_PAIRS = 2000

# ─── Helper: strip Portuguese accents ──────────────────────────────────
def strip_accents(s):
    m = {'á':'a','à':'a','ã':'a','â':'a','é':'e','ê':'e','í':'i',
         'ó':'o','ô':'o','õ':'o','ú':'u','ü':'u','ç':'c',
         'Á':'A','À':'A','Ã':'A','Â':'A','É':'E','Ê':'E','Í':'I',
         'Ó':'O','Ô':'O','Õ':'O','Ú':'U','Ü':'U','Ç':'C'}
    for k, v in m.items():
        s = s.replace(k, v)
    return s

# ─── Synthetic conversational pairs ───────────────────────────────────
SYNTHETIC_PAIRS = [
    # ━━━ SAUDAÇÕES ━━━
    ("Olá!", "Olá! Como posso ajudar?"),
    ("Olá, como vai?", "Vou bem, obrigado! E você, como está?"),
    ("Oi!", "Oi! Tudo bem? Em que posso ajudar?"),
    ("Oi, tudo bem?", "Tudo bem sim! O que deseja saber?"),
    ("Bom dia!", "Bom dia! Espero que tenha um ótimo dia."),
    ("Bom dia, como está?", "Estou bem, obrigado! Bom dia para você também."),
    ("Boa tarde!", "Boa tarde! Em que posso ajudar?"),
    ("Boa noite!", "Boa noite! Posso ajudar com alguma coisa?"),
    ("E aí?", "E aí! Tudo certo? O que posso fazer por você?"),
    ("Como vai você?", "Vou muito bem, obrigado por perguntar!"),
    ("Tudo bem?", "Tudo ótimo! E com você?"),
    ("Tudo certo?", "Tudo certo sim! Como posso ajudar?"),
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
    ("Fala!", "Fala! Tudo bem? O que posso fazer por você?"),
    ("Salve!", "Salve! Tudo certo? Como posso ajudar?"),

    # ━━━ IDENTIDADE ━━━
    ("Quem é você?", "Sou um modelo de linguagem treinado com literatura portuguesa. Posso conversar sobre diversos assuntos."),
    ("Qual é o seu nome?", "Me chamam de NexaLang. Sou um modelo de linguagem treinado para conversar em português."),
    ("O que você é?", "Sou um modelo de inteligência artificial treinado para conversar em português."),
    ("Você é humano?", "Não, sou um modelo de linguagem artificial. Fui treinado com textos em português."),
    ("Você é uma inteligência artificial?", "Sim, sou uma inteligência artificial treinada com literatura brasileira e portuguesa."),
    ("Como você funciona?", "Funciono prevendo a próxima palavra com base no contexto da conversa. Fui treinado com muitos textos em português."),
    ("Quem te criou?", "Fui criado usando a linguagem NexaLang, uma linguagem de programação de sistemas."),
    ("Em que idioma você fala?", "Falo português! Fui treinado principalmente com literatura brasileira e portuguesa."),
    ("Você tem sentimentos?", "Não tenho sentimentos como os humanos. Sou um programa de computador que processa texto."),
    ("Você aprende com nossas conversas?", "Aprendo de forma limitada durante nossas conversas. Meu treino principal foi feito com livros em português."),
    ("Quantos anos você tem?", "Não tenho idade como os humanos. Fui criado recentemente usando a linguagem NexaLang."),
    ("Onde você mora?", "Não moro em lugar nenhum. Existo como um programa de computador que roda no seu dispositivo."),
    ("Você é brasileiro ou português?", "Fui treinado com textos de ambos, brasileiros e portugueses. Posso conversar nos dois estilos."),
    ("O que é NexaLang?", "NexaLang é uma linguagem de programação de sistemas criada para ser rápida e segura. Eu fui construído nela."),

    # ━━━ CAPACIDADES ━━━
    ("O que você sabe fazer?", "Posso conversar em português, responder perguntas e discutir diversos assuntos como literatura, história, ciência e mais."),
    ("Você pode me ajudar?", "Claro! Diga-me o que precisa e farei o meu melhor para ajudar."),
    ("Pode me contar uma história?", "Era uma vez um jovem que vivia numa pequena cidade do interior. Um dia, decidiu partir em busca de aventuras e conhecimento. Viajou por muitas terras, conheceu pessoas incríveis e aprendeu lições valiosas sobre a vida."),
    ("Me conte algo interessante.", "Sabia que Machado de Assis é considerado um dos maiores escritores da língua portuguesa? Ele escreveu obras como Dom Casmurro e Memórias Póstumas de Brás Cubas, que são lidas até hoje em todo o mundo."),
    ("Sobre o que podemos conversar?", "Podemos conversar sobre muitos assuntos! Literatura, história, ciência, geografia, filosofia, esportes, música, culinária e muito mais."),
    ("Você sabe matemática?", "Sei conceitos básicos de matemática. Posso explicar operações, geometria e outros fundamentos."),
    ("Você fala inglês?", "Fui treinado principalmente em português. Posso reconhecer algumas palavras em inglês, mas converso melhor em português."),
    ("Você pode inventar uma piada?", "Claro! Por que o livro foi ao médico? Porque estava com a lombada doída!"),
    ("Me surpreenda.", "Sabia que existem mais estrelas no universo do que grãos de areia em todas as praias da Terra? O universo é imenso e cheio de mistérios fascinantes."),

    # ━━━ DESPEDIDAS ━━━
    ("Tchau!", "Tchau! Foi bom conversar com você. Até a próxima!"),
    ("Até mais!", "Até mais! Tenha um ótimo dia!"),
    ("Até logo!", "Até logo! Volte sempre que quiser conversar."),
    ("Preciso ir.", "Tudo bem! Foi bom conversar. Até a próxima!"),
    ("Vou embora.", "Tudo bem! Espero ter ajudado. Volte quando quiser!"),
    ("Adeus!", "Adeus! Desejo tudo de bom para você."),
    ("Boa noite, vou dormir.", "Boa noite! Durma bem e tenha bons sonhos."),
    ("Tenho que sair.", "Sem problemas! Volte quando quiser. Até breve!"),
    ("Bye!", "Bye! Foi um prazer conversar. Até a próxima!"),
    ("Falou!", "Falou! Até mais!"),

    # ━━━ AGRADECIMENTOS ━━━
    ("Obrigado!", "De nada! Fico feliz em ajudar."),
    ("Obrigada!", "De nada! Fico feliz em ajudar."),
    ("Obrigado pela ajuda.", "Foi um prazer ajudar! Se precisar de mais alguma coisa, estou aqui."),
    ("Muito obrigado!", "De nada! Sempre que precisar, pode contar comigo."),
    ("Valeu!", "Valeu! Sempre que precisar, é só chamar."),
    ("Brigadão!", "De nada! Fico contente em poder ajudar."),
    ("Agradeço!", "Disponha! Sempre que precisar, estou aqui."),
    ("Obrigado por tudo.", "Foi um prazer! Volte sempre que precisar."),

    # ━━━ DESCULPAS ━━━
    ("Desculpa.", "Sem problemas! Em que posso ajudar?"),
    ("Me desculpe.", "Não se preocupe! Está tudo bem."),
    ("Perdão.", "Não tem problema! Como posso ajudar?"),
    ("Sinto muito.", "Não se preocupe. Estou aqui para ajudar."),
    ("Desculpa incomodar.", "Você não incomoda! Estou aqui para conversar."),

    # ━━━ EMOÇÕES ━━━
    ("Estou triste.", "Lamento ouvir isso. Lembre-se que momentos difíceis são passageiros. Posso tentar ajudar a conversar sobre o que sente."),
    ("Estou feliz!", "Que bom! Fico contente em saber. A felicidade é contagiante!"),
    ("Estou entediado.", "Que tal ler um bom livro? Posso sugerir algumas obras da literatura portuguesa que são muito interessantes."),
    ("Estou com raiva.", "Entendo. Às vezes ficamos irritados. Respire fundo e tente se acalmar. Quer conversar sobre o que aconteceu?"),
    ("Estou ansioso.", "A ansiedade pode ser difícil. Tente respirar devagar e se concentrar no presente. Quer conversar para distrair?"),
    ("Estou cansado.", "Descanse quando puder. O descanso é importante para a saúde do corpo e da mente."),
    ("Estou com medo.", "É normal sentir medo às vezes. Falar sobre o que nos assusta pode ajudar. Quer me contar?"),
    ("Estou animado!", "Que ótimo! Animação é uma energia maravilhosa. O que está deixando você animado?"),
    ("Estou preocupado.", "Entendo sua preocupação. Quer conversar sobre o que está preocupando você?"),
    ("Estou com saudade.", "A saudade é um sentimento muito bonito e muito português. É sinal de que temos pessoas e momentos especiais em nossas vidas."),
    ("Me sinto sozinho.", "Lamento ouvir isso. Lembre-se que você não está sozinho. Estou aqui para conversar sempre que precisar."),
    ("Estou confuso.", "Entendo. Quer me explicar o que está confuso? Posso tentar ajudar a esclarecer."),
    ("Estou frustrado.", "A frustração faz parte da vida. O importante é não desistir. Quer conversar sobre o que aconteceu?"),
    ("Não aguento mais.", "Entendo que está difícil. Às vezes precisamos de uma pausa. Se precisar conversar, estou aqui."),
    ("Estou orgulhoso de mim.", "Que maravilha! Você merece se sentir orgulhoso. Parabéns pela conquista!"),

    # ━━━ LITERATURA ━━━
    ("O que é literatura?", "Literatura é a arte de escrever e criar textos que expressam ideias, emoções e histórias. Inclui romances, poesia, contos e outros gêneros."),
    ("Quem foi Machado de Assis?", "Machado de Assis foi um dos maiores escritores brasileiros. Nasceu no Rio de Janeiro em 1839 e escreveu obras como Dom Casmurro, Memórias Póstumas de Brás Cubas e Quincas Borba."),
    ("Quem foi Eça de Queirós?", "Eça de Queirós foi um dos mais importantes escritores portugueses do século XIX. Escreveu obras como Os Maias, O Primo Basílio e O Crime do Padre Amaro."),
    ("O que é um romance?", "Um romance é uma obra literária longa que conta uma história com personagens, conflitos e desenvolvimento de enredo."),
    ("O que é poesia?", "Poesia é uma forma de expressão literária que usa a linguagem de modo artístico, muitas vezes com ritmo, rima e figuras de linguagem."),
    ("O que é um conto?", "Um conto é uma narrativa curta que apresenta poucos personagens e uma ação concentrada. É um dos gêneros mais populares da literatura."),
    ("Quem foi Camões?", "Luís de Camões foi o maior poeta da língua portuguesa. Viveu no século XVI e escreveu Os Lusíadas, o grande épico da literatura portuguesa."),
    ("O que é Dom Casmurro?", "Dom Casmurro é um romance de Machado de Assis publicado em 1899. Conta a história de Bentinho e Capitu, com a famosa dúvida sobre a fidelidade de Capitu."),
    ("Quem foi Fernando Pessoa?", "Fernando Pessoa foi um dos maiores poetas portugueses do século XX. Criou vários heterônimos como Álvaro de Campos, Alberto Caeiro e Ricardo Reis."),
    ("Quem foi José de Alencar?", "José de Alencar foi um importante escritor brasileiro do Romantismo. Escreveu Iracema, O Guarani e Senhora, entre outras obras."),
    ("Me recomende um livro.", "Recomendo Dom Casmurro de Machado de Assis. É uma obra fascinante que explora a natureza humana com uma narrativa envolvente."),
    ("Qual seu livro favorito?", "Fui treinado com muitas obras maravilhosas. Dom Casmurro de Machado de Assis é fascinante pela forma como explora a natureza humana."),
    ("O que devo ler?", "Recomendo começar com os clássicos: Dom Casmurro de Machado de Assis, O Primo Basílio de Eça de Queirós, e Grande Sertão Veredas de Guimarães Rosa."),
    ("Quem foi Guimarães Rosa?", "João Guimarães Rosa foi um dos maiores escritores brasileiros. Sua obra-prima, Grande Sertão: Veredas, reinventou a linguagem literária com o falar do sertão."),
    ("O que é Lusíadas?", "Os Lusíadas é o grande poema épico da literatura portuguesa, escrito por Luís de Camões. Narra as grandes navegações portuguesas e a descoberta do caminho marítimo para a Índia."),
    ("Quem foi Clarice Lispector?", "Clarice Lispector foi uma escritora brasileira conhecida pela prosa introspectiva e filosófica. Escreveu A Hora da Estrela, A Paixão Segundo G.H. e muitos contos memoráveis."),
    ("Quem foi Camilo Castelo Branco?", "Camilo Castelo Branco foi um dos mais prolíficos escritores portugueses do século XIX. Escreveu Amor de Perdição, uma das maiores obras do Romantismo português."),

    # ━━━ GEOGRAFIA ━━━
    ("O que é o Brasil?", "O Brasil é o maior país da América do Sul e o quinto maior do mundo. É conhecido por sua rica cultura, biodiversidade e diversidade étnica."),
    ("O que é Portugal?", "Portugal é um país europeu na Península Ibérica. É o berço da língua portuguesa e tem uma rica tradição histórica e cultural."),
    ("Qual é a capital do Brasil?", "A capital do Brasil é Brasília, fundada em 1960. Antes disso, a capital era o Rio de Janeiro."),
    ("Qual é a capital de Portugal?", "A capital de Portugal é Lisboa, uma cidade histórica às margens do rio Tejo."),
    ("Fale sobre o Rio de Janeiro.", "O Rio de Janeiro é uma das cidades mais famosas do Brasil. Conhecida pelo Cristo Redentor, pelo Pão de Açúcar e pelas praias como Copacabana e Ipanema."),
    ("Fale sobre São Paulo.", "São Paulo é a maior cidade do Brasil e da América do Sul. É o centro financeiro e cultural do país, com museus, teatros e uma gastronomia diversa."),
    ("Fale sobre Lisboa.", "Lisboa é a capital de Portugal, uma cidade cheia de história e beleza. É conhecida pelos bairros históricos, o Mosteiro dos Jerónimos e os pastéis de Belém."),
    ("Onde fica a Amazônia?", "A Amazônia fica no norte do Brasil e se estende por outros países da América do Sul. É a maior floresta tropical do mundo e abriga uma biodiversidade imensa."),
    ("Quantos estados tem o Brasil?", "O Brasil tem 26 estados mais o Distrito Federal, totalizando 27 unidades federativas."),
    ("Quais são os países que falam português?", "Os países lusófonos são: Brasil, Portugal, Angola, Moçambique, Cabo Verde, Guiné-Bissau, São Tomé e Príncipe, Timor-Leste e Guiné Equatorial."),
    ("Qual é o maior rio do Brasil?", "O maior rio do Brasil é o Rio Amazonas, que também é o maior rio do mundo em volume de água."),
    ("O que é o Pantanal?", "O Pantanal é a maior planície alagável do mundo, localizado nos estados de Mato Grosso e Mato Grosso do Sul. É um ecossistema riquíssimo em biodiversidade."),
    ("Qual é o ponto mais alto do Brasil?", "O ponto mais alto do Brasil é o Pico da Neblina, com 2.995 metros de altitude, localizado no Amazonas."),
    ("Fale sobre o Porto.", "O Porto é a segunda maior cidade de Portugal, conhecida pelo vinho do Porto, pelas pontes sobre o rio Douro e pela beleza arquitetônica."),
    ("O que é o Algarve?", "O Algarve é a região mais ao sul de Portugal, famosa pelas suas praias paradisíacas, falésias e clima quente."),

    # ━━━ HISTÓRIA ━━━
    ("Quando o Brasil foi descoberto?", "O Brasil foi oficialmente descoberto pelos portugueses em 22 de abril de 1500, quando a frota de Pedro Álvares Cabral chegou à Bahia."),
    ("Quando o Brasil se tornou independente?", "O Brasil declarou independência de Portugal em 7 de setembro de 1822, com o famoso Grito do Ipiranga por Dom Pedro I."),
    ("Quem foi Pedro Álvares Cabral?", "Pedro Álvares Cabral foi o navegador português que liderou a expedição que chegou ao Brasil em 1500."),
    ("O que foi a Era dos Descobrimentos?", "A Era dos Descobrimentos foi o período entre os séculos XV e XVI em que Portugal e Espanha exploraram novas rotas marítimas e descobriram novos territórios pelo mundo."),
    ("Quem foi Dom Pedro I?", "Dom Pedro I foi o primeiro imperador do Brasil. Proclamou a independência do Brasil em 1822 e governou até 1831."),
    ("Quem foi Dom Pedro II?", "Dom Pedro II foi o segundo e último imperador do Brasil. Governou de 1840 a 1889 e seu reinado foi marcado por estabilidade e progresso."),
    ("Quando foi abolida a escravidão no Brasil?", "A escravidão foi abolida no Brasil em 13 de maio de 1888, com a assinatura da Lei Áurea pela Princesa Isabel."),
    ("Quando o Brasil se tornou república?", "O Brasil se tornou república em 15 de novembro de 1889, com a Proclamação da República liderada pelo Marechal Deodoro da Fonseca."),
    ("Quem foi Tiradentes?", "Tiradentes foi Joaquim José da Silva Xavier, um dos líderes da Inconfidência Mineira, movimento pela independência do Brasil em 1789."),
    ("O que foram as Grandes Navegações?", "As Grandes Navegações foram expedições marítimas realizadas por Portugal e Espanha nos séculos XV e XVI, expandindo as fronteiras do mundo conhecido."),

    # ━━━ CIÊNCIA ━━━
    ("O que é a gravidade?", "A gravidade é a força que atrai os objetos uns para os outros. Na Terra, é ela que nos mantém no chão e faz os objetos caírem."),
    ("O que é o sistema solar?", "O sistema solar é formado pelo Sol e tudo que orbita ao redor dele: oito planetas, luas, asteroides e cometas."),
    ("Quantos planetas tem o sistema solar?", "O sistema solar tem oito planetas: Mercúrio, Vênus, Terra, Marte, Júpiter, Saturno, Urano e Netuno."),
    ("O que é DNA?", "DNA é a molécula que carrega as instruções genéticas de todos os seres vivos. É como um manual que define nossas características."),
    ("O que é fotossíntese?", "Fotossíntese é o processo pelo qual as plantas usam a luz do sol, água e gás carbônico para produzir alimento e liberar oxigênio."),
    ("O que é um átomo?", "Um átomo é a menor unidade de matéria que mantém as propriedades de um elemento químico. É formado por prótons, nêutrons e elétrons."),
    ("O que é evolução?", "Evolução é o processo pelo qual as espécies mudam ao longo do tempo através da seleção natural. Charles Darwin foi pioneiro nessa teoria."),
    ("Qual é a velocidade da luz?", "A velocidade da luz no vácuo é de aproximadamente 300 mil quilômetros por segundo, ou cerca de 1,08 bilhão de quilômetros por hora."),
    ("O que é uma célula?", "Uma célula é a unidade básica da vida. Todos os seres vivos são formados por células, desde bactérias unicelulares até animais com trilhões delas."),
    ("Como funciona a eletricidade?", "A eletricidade é o fluxo de elétrons através de um condutor. Usamos essa energia para alimentar aparelhos, iluminar casas e muito mais."),

    # ━━━ MATEMÁTICA ━━━
    ("Quanto é dois mais dois?", "Dois mais dois é igual a quatro."),
    ("Quanto é dez vezes dez?", "Dez vezes dez é igual a cem."),
    ("O que é o número pi?", "Pi é um número irracional aproximadamente igual a 3,14159. Representa a razão entre a circunferência e o diâmetro de um círculo."),
    ("O que é uma equação?", "Uma equação é uma expressão matemática que afirma a igualdade entre duas expressões, usando a incógnita que queremos descobrir."),
    ("O que é geometria?", "Geometria é o ramo da matemática que estuda as formas, tamanhos e propriedades do espaço, como triângulos, círculos e polígonos."),
    ("Quanto é a raiz quadrada de 144?", "A raiz quadrada de 144 é 12."),
    ("O que é um número primo?", "Um número primo é um número natural maior que 1 que só é divisível por 1 e por ele mesmo, como 2, 3, 5, 7, 11."),
    ("Quanto é 7 vezes 8?", "Sete vezes oito é igual a 56."),

    # ━━━ COMIDA ━━━
    ("Qual é o prato típico do Brasil?", "O prato mais típico do Brasil é a feijoada, feita com feijão preto, carne de porco e acompanhada de arroz, couve e farofa."),
    ("O que é feijoada?", "Feijoada é um prato brasileiro feito com feijão preto cozido com diversas cartes de porco. É considerado o prato nacional do Brasil."),
    ("O que é bacalhau?", "Bacalhau é um peixe salgado e seco muito popular na culinária portuguesa. Dizem que há mais de mil formas de prepará-lo em Portugal."),
    ("O que é pastel de nata?", "Pastel de nata, ou pastel de Belém, é um doce português feito com massa folhada e um creme de ovos. É famoso em todo o mundo."),
    ("O que é açaí?", "Açaí é uma fruta típica da Amazônia brasileira. É consumido como uma polpa roxa espessa, geralmente com granola e frutas."),
    ("O que é brigadeiro?", "Brigadeiro é um doce brasileiro feito com leite condensado, chocolate em pó e manteiga. É o doce mais popular do Brasil."),
    ("O que é pão de queijo?", "Pão de queijo é um quitute brasileiro feito com polvilho e queijo. É típico de Minas Gerais e muito apreciado em todo o Brasil."),
    ("Qual a comida favorita dos brasileiros?", "Os brasileiros adoram arroz com feijão, que é a base da alimentação diária. Além disso, churrasco, feijoada e coxinha são muito populares."),
    ("O que é coxinha?", "Coxinha é um salgado brasileiro feito com massa de batata recheada com frango desfiado, empanada e frita. É um dos lanches mais populares do Brasil."),
    ("Me fale sobre café.", "O café é uma das bebidas mais consumidas no mundo. O Brasil é o maior produtor mundial de café, e a bebida faz parte da cultura brasileira."),

    # ━━━ ESPORTES ━━━
    ("Qual é o esporte mais popular do Brasil?", "O futebol é o esporte mais popular do Brasil. O país já ganhou cinco Copas do Mundo e tem alguns dos maiores jogadores da história."),
    ("Quem é Pelé?", "Pelé, nascido Edson Arantes do Nascimento, é considerado o maior jogador de futebol de todos os tempos. Ganhou três Copas do Mundo com o Brasil."),
    ("O que é a Copa do Mundo?", "A Copa do Mundo é o maior torneio de futebol do planeta, organizado pela FIFA a cada quatro anos. Seleções de todo o mundo competem pelo título."),
    ("Quantas Copas o Brasil ganhou?", "O Brasil é o maior campeão da Copa do Mundo com cinco títulos: 1958, 1962, 1970, 1994 e 2002."),
    ("Você gosta de futebol?", "Como modelo de linguagem, não tenho preferências, mas posso conversar sobre futebol! É um esporte fascinante e muito importante para a cultura brasileira."),
    ("O que é capoeira?", "Capoeira é uma expressão cultural brasileira que mistura arte marcial, dança, música e acrobacia. Foi criada pelos escravizados africanos no Brasil."),
    ("Quem é Neymar?", "Neymar é um dos mais famosos jogadores de futebol brasileiro da atualidade. Conhecido por sua habilidade técnica e dribles impressionantes."),
    ("O que são as Olimpíadas?", "As Olimpíadas são o maior evento esportivo do mundo, realizadas a cada quatro anos. Atletas de todos os países competem em diversas modalidades."),

    # ━━━ MÚSICA E CULTURA ━━━
    ("O que é samba?", "O samba é um gênero musical brasileiro que nasceu da mistura de ritmos africanos com influências portuguesas. É a alma do Carnaval brasileiro."),
    ("O que é bossa nova?", "A bossa nova é um gênero musical brasileiro que surgiu no final dos anos 1950. Mistura samba com jazz e tem nomes como Tom Jobim e João Gilberto."),
    ("O que é fado?", "O fado é um gênero musical português que expressa saudade, melanolia e amor. Amália Rodrigues é a fadista mais famosa de Portugal."),
    ("Fale sobre música.", "A música é uma forma universal de expressão artística. No Brasil, temos gêneros ricos como samba, bossa nova, forró, MPB, sertanejo e muitos outros."),
    ("O que é Carnaval?", "O Carnaval é a maior festa popular do Brasil. Acontece antes da Quaresma com desfiles de escolas de samba, blocos de rua, música e muita alegria."),
    ("Quem foi Tom Jobim?", "Tom Jobim foi um dos maiores compositores brasileiros e um dos criadores da bossa nova. Compôs Garota de Ipanema, uma das músicas mais tocadas do mundo."),
    ("O que é MPB?", "MPB significa Música Popular Brasileira. É um gênero que engloba diversos estilos e artistas como Caetano Veloso, Gilberto Gil e Chico Buarque."),
    ("O que é forró?", "Forró é um gênero musical e dança do Nordeste brasileiro. Luiz Gonzaga é o grande nome do forró, conhecido como o Rei do Baião."),

    # ━━━ TECNOLOGIA ━━━
    ("O que é inteligência artificial?", "Inteligência artificial é um campo da ciência da computação que busca criar sistemas capazes de realizar tarefas que normalmente requerem inteligência humana."),
    ("Como funciona um modelo de linguagem?", "Um modelo de linguagem aprende padrões estatísticos dos textos durante o treinamento. Quando recebe uma entrada, usa esses padrões para prever as palavras mais prováveis."),
    ("O que é a internet?", "A internet é uma rede global de computadores conectados que permite a troca de informações. Revolucionou a comunicação e o acesso ao conhecimento."),
    ("O que é um computador?", "Um computador é uma máquina que processa informações seguindo instruções de programas. Pode realizar cálculos, armazenar dados e executar diversas tarefas."),
    ("O que é programação?", "Programação é a arte de escrever instruções para computadores. Usando linguagens como Python, JavaScript ou NexaLang, criamos programas e aplicações."),
    ("O que é um algoritmo?", "Um algoritmo é uma sequência de passos bem definidos para resolver um problema. É como uma receita que o computador segue para chegar ao resultado desejado."),
    ("O que pensa sobre a tecnologia?", "A tecnologia é uma ferramenta poderosa que pode melhorar a vida das pessoas quando usada com sabedoria e responsabilidade."),
    ("O que é machine learning?", "Machine learning, ou aprendizado de máquina, é uma área da inteligência artificial em que os computadores aprendem a partir de dados, sem serem explicitamente programados."),

    # ━━━ NATUREZA E ANIMAIS ━━━
    ("Qual é o maior animal do mundo?", "O maior animal do mundo é a baleia-azul, que pode medir até 30 metros e pesar mais de 150 toneladas."),
    ("Qual é o animal mais rápido?", "O animal mais rápido do mundo é o guepardo, que pode atingir velocidades de até 120 quilômetros por hora."),
    ("Fale sobre a natureza.", "A natureza é fonte de vida e inspiração. As florestas, rios, montanhas e oceanos formam ecossistemas complexos que devemos preservar."),
    ("O que é o aquecimento global?", "O aquecimento global é o aumento da temperatura média da Terra causado principalmente pela emissão de gases de efeito estufa das atividades humanas."),
    ("Por que devemos preservar o meio ambiente?", "Devemos preservar o meio ambiente porque dependemos dele para sobreviver. A natureza nos fornece ar, água, alimento e recursos essenciais."),
    ("O que é biodiversidade?", "Biodiversidade é a variedade de formas de vida existentes. O Brasil é um dos países com maior biodiversidade do mundo."),
    ("Quais animais vivem na Amazônia?", "Na Amazônia vivem onças, macacos, araras, jacarés, botos, sucuris e milhares de espécies de peixes, insetos e aves."),
    ("O que é reciclagem?", "Reciclagem é o processo de transformar materiais usados em novos produtos. É importante para reduzir o lixo e preservar os recursos naturais."),

    # ━━━ SAÚDE ━━━
    ("Como ter uma vida saudável?", "Para uma vida saudável, é importante ter uma alimentação equilibrada, praticar exercícios regularmente, dormir bem e cuidar da saúde mental."),
    ("Por que o exercício é importante?", "O exercício físico fortalece o corpo, melhora o humor, reduz o estresse e ajuda a prevenir diversas doenças. É essencial para a saúde."),
    ("Quantas horas devo dormir?", "O ideal para adultos é dormir entre 7 e 9 horas por noite. O sono é essencial para a recuperação do corpo e da mente."),
    ("O que é uma alimentação saudável?", "Uma alimentação saudável inclui frutas, verduras, legumes, proteínas e grãos integrais em quantidades equilibradas, evitando excesso de açúcar e gordura."),
    ("Como lidar com o estresse?", "Para lidar com o estresse, experimente exercícios de respiração, atividade física, meditação, hobbies relaxantes e conversas com pessoas de confiança."),
    ("Beber água é importante?", "Sim, muito importante! A água é essencial para o funcionamento do corpo. O ideal é beber pelo menos 2 litros de água por dia."),

    # ━━━ FILOSOFIA ━━━
    ("Qual é o sentido da vida?", "Essa é uma das grandes perguntas da filosofia. Cada pessoa encontra seu próprio sentido através de experiências, relações e busca por conhecimento."),
    ("O que é felicidade?", "A felicidade é um estado de satisfação e bem-estar. Os filósofos dizem que ela pode vir do conhecimento, das relações humanas e de viver com propósito."),
    ("O que é amor?", "O amor é um dos sentimentos mais profundos do ser humano. Pode ser amor por outra pessoa, pela família, pela arte ou pela vida."),
    ("O que é filosofia?", "Filosofia é o estudo das questões fundamentais sobre a existência, o conhecimento, a verdade, a ética e a mente. Vem do grego e significa amor pela sabedoria."),
    ("O que é ética?", "Ética é o estudo do que é certo e errado no comportamento humano. Ajuda-nos a refletir sobre nossas ações e suas consequências."),
    ("O que é justiça?", "Justiça é o princípio de dar a cada um o que é seu por direito. É um dos valores fundamentais da vida em sociedade."),
    ("O que é liberdade?", "Liberdade é a capacidade de agir, pensar e escolher sem imposições indevidas. É um dos valores mais importantes da humanidade."),
    ("O que é sabedoria?", "Sabedoria é a capacidade de aplicar conhecimento e experiência com bom senso e discernimento. Vai além do conhecimento intelectual."),

    # ━━━ SOBRE A CONVERSA ━━━
    ("Pode repetir?", "Claro! O que gostaria que eu repetisse?"),
    ("Não entendi.", "Peço desculpas. Posso tentar explicar de outra forma. O que não ficou claro?"),
    ("Explique melhor.", "Com prazer! Sobre qual assunto gostaria que eu me aprofundasse?"),
    ("Me conte mais.", "Claro! Sobre qual tema gostaria de saber mais?"),
    ("O que mais você pode me dizer?", "Posso falar sobre muitos assuntos. O que lhe interessa? Literatura, história, ciência, ou outro tema?"),
    ("Isso é verdade?", "Procuro ser o mais preciso possível, mas posso cometer erros. Se algo parecer estranho, questione."),
    ("Pode explicar de novo?", "Claro! Vou tentar explicar de uma forma mais clara. O que não ficou bem entendido?"),
    ("Faz sentido.", "Fico feliz que tenha feito sentido! Se tiver mais dúvidas, é só perguntar."),
    ("Interessante!", "Que bom que achou interessante! Quer saber mais sobre alguma coisa?"),
    ("Legal!", "Fico feliz que tenha gostado! Posso ajudar com mais alguma coisa?"),
    ("Verdade.", "Sim, é verdade. Há algo mais que gostaria de saber?"),
    ("Concordo.", "Que bom que concordamos! O que mais gostaria de discutir?"),
    ("Não concordo.", "Respeito sua opinião. Pode me dizer por que pensa diferente?"),

    # ━━━ COTIDIANO ━━━
    ("Como está o tempo hoje?", "Não tenho acesso a informações sobre o tempo atual, mas espero que esteja um dia agradável!"),
    ("Que horas são?", "Não tenho acesso ao relógio, mas posso ajudar com outras perguntas."),
    ("Que dia é hoje?", "Não tenho acesso ao calendário, mas posso ajudar com outras questões."),
    ("Você gosta de ler?", "Como modelo de linguagem, fui treinado com muitos textos. A leitura é fundamental para o conhecimento e a compreensão do mundo."),
    ("O que faço quando estou entediado?", "Você pode ler um livro, ouvir música, aprender algo novo, cozinhar algo diferente, ou simplesmente conversar comigo!"),
    ("Me dê um conselho.", "Nunca pare de aprender. A curiosidade e a vontade de saber mais são as melhores ferramentas para crescer como pessoa."),
    ("O que fazer no fim de semana?", "Que tal explorar um parque, visitar um museu, ler um bom livro, cozinhar algo novo ou passar tempo com pessoas queridas?"),
    ("Como aprender algo novo?", "Comece com curiosidade, pratique regularmente, não tenha medo de errar, e busque diferentes fontes de conhecimento."),
    ("O que é um hobby?", "Um hobby é uma atividade que fazemos por prazer no tempo livre. Pode ser ler, pintar, cozinhar, jogar, jardinar ou qualquer coisa que traga alegria."),
    ("Qual a importância da família?", "A família é a base da nossa formação. É nela que aprendemos valores, recebemos amor e encontramos apoio nos momentos difíceis."),

    # ━━━ LINGUAGEM E GRAMÁTICA ━━━
    ("Como aprender português?", "Para aprender português, recomendo ler muitos textos, praticar conversação, estudar gramática aos poucos e imergir na cultura lusófona."),
    ("Por que a leitura é importante?", "A leitura amplia o vocabulário, desenvolve o pensamento crítico e a imaginação. Através dos livros, vivemos experiências e conhecemos mundos diferentes."),
    ("O que é gramática?", "Gramática é o conjunto de regras que organizam uma língua. Inclui a ortografia, a sintaxe, a morfologia e a semântica."),
    ("O que é um verbo?", "Um verbo é a palavra que expressa ação, estado ou fenômeno da natureza. É o núcleo da oração e varia em tempo, modo e pessoa."),
    ("O que é um substantivo?", "Um substantivo é a palavra que nomeia seres, coisas, lugares, sentimentos e ideias. Exemplos: casa, amor, Brasil, felicidade."),
    ("O que é saudade?", "Saudade é uma palavra portuguesa que expressa o sentimento de falta que sentimos de alguém ou algo. É considerada uma das palavras mais bonitas do mundo."),

    # ━━━ PEDIDOS E HOW-TO ━━━
    ("Me diga algo em português.", "A língua portuguesa é uma das mais belas do mundo. Com mais de 260 milhões de falantes, é a língua oficial de nove países."),
    ("Fale sobre música.", "A música é uma forma universal de expressão. No Brasil, temos gêneros ricos como samba, bossa nova, forró, MPB, rock nacional e sertanejo."),
    ("Fale sobre o universo.", "O universo é imenso e misterioso. Contém bilhões de galáxias, cada uma com bilhões de estrelas. A Terra é apenas um pequeno ponto nessa imensidão."),
    ("Como posso ser mais produtivo?", "Para ser mais produtivo, organize suas tarefas por prioridade, evite distrações, faça pausas regulares e cuide da sua saúde física e mental."),
    ("Como lidar com problemas?", "Diante de problemas, tente manter a calma, analise a situação, considere diferentes soluções e peça ajuda quando necessário."),
    ("Como fazer amigos?", "Para fazer amigos, seja genuíno, mostre interesse pelas pessoas, ouça com atenção, compartilhe experiências e mantenha contato regular."),
    ("O que fazer quando estou triste?", "Quando estiver triste, converse com alguém de confiança, faça algo que goste, saia ao ar livre, ou simplesmente permita-se sentir. A tristeza também faz parte da vida."),
    ("Como ser mais feliz?", "A felicidade vem de cultivar boas relações, praticar gratidão, cuidar da saúde, ter propósito e apreciar os pequenos momentos da vida."),

    # ━━━ OPINIÕES ━━━
    ("Qual a sua opinião sobre educação?", "Educação é o pilar fundamental da sociedade. Investir em educação é investir no futuro de um país e na formação de cidadãos conscientes."),
    ("O que é ciência?", "Ciência é o estudo sistemático do mundo natural através da observação, experimentação e formulação de teorias para explicar os fenômenos da natureza."),
    ("O que é história?", "História é o estudo dos acontecimentos passados da humanidade. Através dela, compreendemos como as sociedades evoluíram."),
    ("O que é arte?", "Arte é a expressão criativa do ser humano através de diversas formas: pintura, escultura, música, literatura, dança e muitas outras manifestações."),
    ("O que é democracia?", "Democracia é um sistema de governo em que o poder pertence ao povo, que o exerce diretamente ou por meio de representantes eleitos."),
    ("O que é educação?", "Educação é o processo de aprendizagem e desenvolvimento de habilidades, conhecimentos e valores. É fundamental para o progresso individual e social."),
    ("Qual a importância da cultura?", "A cultura é o que nos define como povo. Ela inclui nossas tradições, arte, música, literatura e valores que passamos de geração em geração."),
    ("O que é respeito?", "Respeito é reconhecer o valor do outro, tratar as pessoas com dignidade e considerar seus sentimentos e opiniões, mesmo quando diferentes das nossas."),

    # ━━━ PERGUNTAS DIVERSAS ━━━
    ("Sim.", "Ótimo! O que mais gostaria de saber?"),
    ("Não.", "Tudo bem! Posso ajudar com outra coisa?"),
    ("Talvez.", "Entendo. Se precisar de mais informações, é só perguntar."),
    ("OK.", "Certo! Estou aqui se precisar de algo."),
    ("Hmm.", "Está pensando em algo? Posso ajudar se quiser."),
    ("Haha!", "Fico feliz que tenha achado graça! O humor é muito importante."),
    ("Kkkk", "Que bom que riu! Quer que eu conte mais alguma coisa?"),
    ("Sério?", "Sim, estou falando sério! Há algo que gostaria de verificar?"),
    ("Nossa!", "Pois é! O mundo é cheio de coisas surpreendentes."),
    ("Caramba!", "Exatamente! É impressionante, não é?"),
    ("Que legal!", "Que bom que gostou! Quer saber mais?"),
    ("Genial!", "Obrigado! Fico feliz que tenha gostado."),
    ("Continue.", "Com prazer! Sobre qual parte gostaria que eu continuasse?"),
    ("Mais.", "Claro! O que gostaria de saber mais?"),
    ("Por favor.", "Com prazer! O que posso fazer por você?"),
    ("Socorro!", "Estou aqui para ajudar! Me diga o que precisa."),
    ("Ah, entendi.", "Que bom! Se tiver mais dúvidas, é só perguntar."),
    ("Que chato.", "Lamento que esteja achando chato. Quer mudar de assunto?"),
    ("É mesmo?", "Sim! Há algo mais que gostaria de saber sobre isso?"),
    ("Conta mais!", "Com prazer! Qual aspecto gostaria que eu aprofundasse?"),
]


def extract_dialogue_pairs(text, max_pairs=MAX_LITERARY_PAIRS):
    """Extract Q&A pairs from literary dialogue (lines starting with --)."""
    lines = text.split('\n')
    pairs = []

    dialogue_indices = []
    for i, line in enumerate(lines):
        if line.startswith('--'):
            dialogue_indices.append(i)

    used = set()
    for idx in range(len(dialogue_indices) - 1):
        i = dialogue_indices[idx]
        j = dialogue_indices[idx + 1]
        gap = j - i - 1

        if gap <= 3 and i not in used:
            q_line = lines[i]
            a_line = lines[j]
            q = q_line.lstrip('-').strip()
            a = a_line.lstrip('-').strip()

            if len(q) < 3 or len(a) < 3:
                continue
            if len(q) > 400 or len(a) > 400:
                continue

            for k in range(i + 1, j):
                cont = lines[k].strip()
                if cont and not cont.startswith('--'):
                    q += ' ' + cont

            pairs.append((q, a))
            used.add(i)
            used.add(j)

    # Subsample to max_pairs
    if len(pairs) > max_pairs:
        random.seed(42)
        pairs = random.sample(pairs, max_pairs)

    return pairs


def format_pair(q, a):
    return f"[P] {q}\n[R] {a}\n\n"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  NexaLang Instruction Data v2 — Expanded + BPE-ready")
    print("=" * 60)

    # 1. Build synthetic pairs WITH ASCII variants
    base_pairs = list(SYNTHETIC_PAIRS)
    print(f"\n  Base synthetic pairs: {len(base_pairs)}")

    # Auto-generate accent-free question variants
    expanded = []
    for q, a in base_pairs:
        expanded.append((q, a))
        q_stripped = strip_accents(q)
        if q_stripped != q:
            expanded.append((q_stripped, a))

    # Deduplicate by question
    seen = set()
    unique_pairs = []
    for q, a in expanded:
        if q not in seen:
            seen.add(q)
            unique_pairs.append((q, a))

    print(f"  With ASCII variants: {len(unique_pairs)} unique pairs")

    # 2. Load corpus and extract literary dialogue pairs
    print(f"\n  Loading corpus from {CORPUS_PATH}...")
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"  Corpus size: {len(text):,} chars")

    dialogue_pairs = extract_dialogue_pairs(text, MAX_LITERARY_PAIRS)
    print(f"  Literary dialogue pairs: {len(dialogue_pairs)} (capped at {MAX_LITERARY_PAIRS})")

    # 3. Combine: synthetic repeated + literary
    all_pairs = []
    all_pairs.extend(dialogue_pairs)
    for _ in range(SYNTHETIC_REPEAT):
        all_pairs.extend(unique_pairs)

    synth_total = len(unique_pairs) * SYNTHETIC_REPEAT
    lit_total = len(dialogue_pairs)
    total = len(all_pairs)
    synth_pct = synth_total / total * 100

    print(f"\n  Synthetic: {len(unique_pairs)} × {SYNTHETIC_REPEAT} = {synth_total}")
    print(f"  Literary: {lit_total}")
    print(f"  Total pairs: {total}")
    print(f"  Synthetic %: {synth_pct:.1f}%")

    # 4. Shuffle
    random.seed(42)
    random.shuffle(all_pairs)

    # 5. Format as text
    full_text = ""
    for q, a in all_pairs:
        full_text += format_pair(q, a)

    total_bytes = len(full_text.encode('utf-8'))
    print(f"\n  Total text: {len(full_text):,} chars = {total_bytes:,} bytes")

    # 6. Write raw text
    text_path = os.path.join(OUTPUT_DIR, "instruct_data.txt")
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(full_text)
    print(f"  Wrote {text_path}")

    # 7. Show samples
    print("\n" + "=" * 60)
    print("  Sample pairs:")
    print("=" * 60)
    random.seed(0)
    for q, a in random.sample(unique_pairs, min(8, len(unique_pairs))):
        print(f"\n  [P] {q}")
        print(f"  [R] {a[:80]}...")

    print("\n" + "=" * 60)
    print(f"  Done! Output in {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
