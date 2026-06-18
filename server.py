import json
import time
import random
import asyncio
import uvicorn
import datetime
import traceback
import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI, WebSocket, WebSocketDisconnect


# Global variables.
NUM_SYMBOLS = 4    # ATCG
NUM_DIGITS = 5    # some integers comes from a string of the form 'X_1 X_2...X_n' where each X_i is one of the symbols (above) and has NUM_DIGITS digits
NUM_DIGITS_2 = 2
NUM_DIGITS_3 = 6
NUM_DIGITS_4 = 4
A_MIN, A_MAX = 4, 7
B_MIN, B_MAX = 4, 7
C_MIN, C_MAX = 4, 7
NUM_INPUTS = 16
NUM_OUTPUTS = 4
MAPPING = {0: 0.2, 1: 0.4, 2: 0.6, 3: 0.8}
ATCG_MAPPING = {0: 'A', 1: 'T', 2: 'C', 3: 'G'}
SHOW = True
SHOW_ATCG = False
NPC = {}

# Variáveis para salva atividades neuronais.
it = -1
it_real = -1
blob_history = []
num_connections_history = []
N = 100    # salva o estado apenas a cada N iterações (mas as primeiras N iterações entram no histórico)
start = None

with open('blob_history.txt', 'w') as f:
    f.write('')

# Inicia a API.
app = FastAPI()
server = None


# Endpoint WebSocket: o Godot se conecta aqui. Execute o script com python server.py.
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()  # Aceita a conexão do cliente
    print("✅ Conexão WebSocket estabelecida")
    global blob_history

    try:        
        while True:
            try:
                data = await websocket.receive_text()

            except WebSocketDisconnect:
                print("Cliente desconectou.")
                break

            except Exception as e:
                print("Erro recebendo mensagem:", repr(e))
                traceback.print_exc()
                continue
                
            payload = json.loads(data)  # Converte a mensagem para JSON
            npcs = payload.get("batch", [])  # Obtém os dados dos NPCs
            results = []  # Lista para armazenar os resultados     
 
            for npc in npcs:
                npc_action = npc.get("npc_action")    # process ou gen_brain
                npc_id = npc.get("id")  # Obtém o ID do NPC
                inputs = npc.get("inputs")  # Obtém os inputs do NPC
                               
                if npc_action in ["gen_brain", "process"]:                               
                    if npc_id is None:
                        continue
                    
                    int_id = int(npc_id.replace('blob_', ''))

                    if inputs is not None:                 
                        if len(inputs) != NUM_INPUTS:
                            print("Número de inputs do server e engine estão diferentes.")
                            server.should_exit = True
                                        
                    if npc_action == "gen_brain":
                        gens, gens_num = gen_brain()  # Gera o cérebro simulado
                        gens_num["id"] = npc_id  # Mantém a associação do blob
                        results.append({"id": gens_num["id"], 
                                        "npc_action": "gen_brain"})  # Adiciona o resultado à lista
                        NPC[npc_id] = gens_num  # Adiciona ao dicionário global de cérebros
                        NPC[npc_id]['gens'] = gens  # Memoriza o código genético puro
                        extras(npc_id)  # Inclui mais elementos necessários para os processamentos
                    elif npc_action == "process" and npc_id in NPC:
                        outputs = process_npc(npc_id, inputs)  # Processa os inputs e determina a ação
                        results.append({"id": npc_id, 
                                        "outputs": outputs, 
                                        "size": A_MAX * B_MAX * C_MAX, 
                                        "num_connections": NPC[npc_id]['num_connections'], 
                                        "npc_action": "process"})  # Adiciona o resultado à lista
                elif npc_action == "offspring":
                    npc_id1 = npc.get("id1")
                    npc_id2 = npc.get("id2")
                    gens, gens_num = offspring(npc_id1, npc_id2) 
                    
                    if gens == '': 
                        results.append({"id": "", "npc_action": "offspring"}) 
                    else:
                        gens_num["id"] = "blob_" + str(int(len(NPC) + np.random.randint(1000)))
                        npc_id = gens_num["id"]
                        results.append({"id": gens_num["id"], "npc_action": "offspring"}) 
                        NPC[npc_id] = gens_num  
                        NPC[npc_id]['gens'] = gens  # Memoriza o código genético puro
                        extras(npc_id)  
                   
            response = json.dumps({"results": results})  # Cria a resposta com os resultados
            await websocket.send_text(response)  # Envia a resposta ao cliente
            
    except Exception as e:
        print("⚠️ Erro na conexão WebSocket:", e)
        traceback.print_exc()

    finally:
        # Este bloco executa mesmo em desconexão normal
        print("💾 Salvando históricos...")
        #with open("blob_history.txt", "w") as f:
        #    f.writelines(blob_history)
        #with open("num_connections_history.txt", "w") as f:
        #    f.writelines(num_connections_history)
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()
            print("🔒 Conexão WebSocket encerrada.")

# Função que gera um cérebro simulado. 
def gen_brain():
    success = False
    
    while not success:
        # Dictionary containing all genes in symbolic form.
        gens = {}

        # Dictionary containing all genes in numeric form.
        gens_num = {}

        ######################################################################################################
        # Dimensions.
        gens['dims'] = np.random.randint(NUM_SYMBOLS, size=[3, NUM_DIGITS])

        digits = gens['dims'][0, :]
        a = max(A_MIN, digits2int(digits) % (A_MAX + 1))

        digits = gens['dims'][1, :]
        b = max(B_MIN, digits2int(digits) % (B_MAX + 1))

        digits = gens['dims'][2, :]
        c = max(C_MIN, digits2int(digits) % (C_MAX + 1))

        gens_num['dims'] = [a, b, c]
        num_neurons = a * b * c
        
        if SHOW:
            print(f'Dimensions = {a} x {b} x {c}    Number of neurons = {num_neurons}')

        ######################################################################################################
        # Connections with inputs.
        gens['inputs'] = np.random.randint(NUM_SYMBOLS, size=[NUM_INPUTS, 3 * NUM_DIGITS])
        gens_num['inputs'] = []

        for n in range(NUM_INPUTS):
            i = digits2int(gens['inputs'][n, : NUM_DIGITS]) % a
            j = digits2int(gens['inputs'][n, NUM_DIGITS : 2*NUM_DIGITS]) % b
            k = digits2int(gens['inputs'][n, 2*NUM_DIGITS : 3*NUM_DIGITS]) % c   
            gens_num['inputs'].append(i + j*a + k*a*b)

        gens_num['inputs'] = list(set(gens_num['inputs']))
        num_connections_input_neuron = len(gens_num['inputs'])
        
        if num_connections_input_neuron != NUM_INPUTS:
            if SHOW:
                print('É necessário repetir a operação pois existem inputs não conectados a neurônios.')
            continue
        
        if SHOW:
            print(f"Neurons connected to inputs: {gens_num['inputs']}    Number of connections: {num_connections_input_neuron}")

        ######################################################################################################
        # Connections with outputs.
        gens['outputs'] = np.random.randint(NUM_SYMBOLS, size=[NUM_OUTPUTS, 3 * NUM_DIGITS])
        gens_num['outputs'] = []

        for n in range(NUM_OUTPUTS):
            i = digits2int(gens['outputs'][n, : NUM_DIGITS]) % a
            j = digits2int(gens['outputs'][n, NUM_DIGITS : 2*NUM_DIGITS]) % b
            k = digits2int(gens['outputs'][n, 2*NUM_DIGITS : 3*NUM_DIGITS]) % c
            gens_num['outputs'].append(i + j*a + k*a*b)

        gens_num['outputs'] = list(set(gens_num['outputs']))
        num_connections_output_neuron = len(gens_num['outputs'])
        
        if num_connections_output_neuron != NUM_OUTPUTS:
            if SHOW:
                print('É necessário repetir a operação pois existem outputs não conectados a neurônios.')
            continue

        for x in gens_num['outputs']:
            if x in gens_num['inputs']:
                if SHOW:
                    print(f'É necessário repetir a operação pois um neurônio conectado ao output já está sendo usado por um de input: {x}')
                success = False
                break
            success = True
            
        if not success:
            continue                

        if SHOW:
            print(f"Neurons connected to outputs: {gens_num['outputs']}    Number of connections: {num_connections_output_neuron}")
            
        ######################################################################################################
        # Output thresholds to make action.
        gens['outputs_thr'] = np.random.randint(NUM_SYMBOLS, size=[len(gens['outputs'])])
        gens_num['outputs_thr'] = {}
        w = 0

        for x in gens_num['outputs']:
            gens_num['outputs_thr'][x] = MAPPING[gens['outputs_thr'][w]]
            w += 1

        if SHOW:
            print(f"Output threshold of neuron {x} = {gens_num['outputs_thr'][x]}, ...")

        ######################################################################################################
        # Region sizes.
        gens['regions'] = np.random.randint(NUM_SYMBOLS, size=[num_neurons, 3 * NUM_DIGITS_2])
        gens_num['regions'] = {}
        w = 0

        for i in range(a):
            for j in range(b):
                for k in range(c):
                    region_size_x = digits2int(gens['regions'][w, : NUM_DIGITS_2]) % a
                    region_size_y = digits2int(gens['regions'][w, NUM_DIGITS_2 : 2*NUM_DIGITS_2]) % b
                    region_size_z = digits2int(gens['regions'][w, 2*NUM_DIGITS_2 : 3*NUM_DIGITS_2]) % c
                    gens_num['regions'][i + j*a + k*a*b] = [region_size_x, region_size_y, region_size_z]
                    w += 1
            
        ######################################################################################################
        # Conexões pré-programadas no cérebro. Cada neurônio está conectado a um conjunto de neurônios diferentes no início.
        gens['num_initial_connections'] = np.random.randint(NUM_SYMBOLS, size=[NUM_DIGITS_4])
        num_initial_connections = digits2int(gens['num_initial_connections']) % (num_neurons - 1)
        gens_num['num_initial_connections'] = num_initial_connections
        gens['connections'] = {}
        gens_num['connections'] = {}

        for i in range(a):
            for j in range(b):
                for k in range(c):
                    idx = i + j*a + k*a*b
                    gens['connections'][idx] = np.random.randint(NUM_SYMBOLS, size=[num_initial_connections, 3 * NUM_DIGITS_2])
                    gens_num['connections'][i + j*a + k*a*b] = []
                    
                    for w in range(num_initial_connections):
                        neuron_x = digits2int(gens['connections'][idx][w, : NUM_DIGITS_2]) % a
                        neuron_y = digits2int(gens['connections'][idx][w, NUM_DIGITS_2 : 2*NUM_DIGITS_2]) % b
                        neuron_z = digits2int(gens['connections'][idx][w, 2*NUM_DIGITS_2 : 3*NUM_DIGITS_2]) % c  
                        neuron2 = neuron_x + neuron_y*a + neuron_z*a*b
                        
                        # Verifica se não é conexão repetida.
                        if neuron2 not in gens_num['connections'][idx]:
                            # Verifica se já não há uma conexão na direção contrária.
                            if (neuron2 not in gens_num['connections']) or (idx not in gens_num['connections'][neuron2 ]):
                                # Verifica se o neurônio candidato está dentro da região permitida.
                                radius_x, radius_y, radius_z = gens_num['regions'][idx]
                                if i - radius_x <= neuron_x <= i + radius_x:
                                    if j - radius_y <= neuron_y <= j + radius_y:
                                        if k - radius_z <= neuron_z <= k + radius_z:
                                            gens_num['connections'][idx].append(neuron2)

        if SHOW:
            print(f"Number of initial connections per neuron = {gens_num['num_initial_connections']}")

        ######################################################################################################
        # Probabilities of making connections.
        gens['probs'] = np.random.randint(NUM_SYMBOLS, size=[num_neurons])
        gens_num['probs'] = {}
        w = 0

        for i in range(a):
            for j in range(b):
                for k in range(c):
                    gens_num['probs'][i + j*a + k*a*b] = MAPPING[gens['probs'][w]]
                    w += 1

        ######################################################################################################
        # Threshold per neuron.
        gens['thr'] = np.random.randint(NUM_SYMBOLS, size=[num_neurons])
        gens_num['thr'] = {}
        w = 0

        for i in range(a):
            for j in range(b):
                for k in range(c):
                    gens_num['thr'][i + j*a + k*a*b] = MAPPING[gens['thr'][w]]
                    w += 1

        ######################################################################################################
        # Decay of neuron connections.
        gens['decay'] = np.random.randint(NUM_SYMBOLS, size=[NUM_DIGITS_3])
        gens_num['decay'] = float('0.' + ''.join([str(x) for x in gens['decay']]))
        
        if SHOW:
            print(f"Decay e = {gens_num['decay']}")

        # Number of iterations that the neuron is inactive after sending a signal.
        gens['inactive_iter'] = np.random.randint(NUM_SYMBOLS, size=[NUM_DIGITS_2])
        gens_num['inactive_iter'] = digits2int(gens['inactive_iter']) + 1
        
        if SHOW:
            print(f"Number of inactive iterations after sending a signal = {gens_num['inactive_iter']}")
            
        # Exponential factor saturation response of the signals.
        gens['exp_response'] = np.random.randint(NUM_SYMBOLS, size=[NUM_DIGITS_2])
        gens_num['exp_response'] = digits2int(gens['exp_response'])//10 + 1
        
        if SHOW:
            print(f"Exponential factor saturation response of the signals = {gens_num['exp_response']}")

        # Number of iterations that the neuron is weakened after sending a signal.
        gens['weakened_iter'] = np.random.randint(NUM_SYMBOLS, size=[NUM_DIGITS_2])
        gens_num['weakened_iter'] = gens_num['inactive_iter'] + digits2int(gens['weakened_iter']) + 1
        
        if SHOW:
            print(f"Number of weakened iterations after sending a signal = {gens_num['weakened_iter']}")
        
        ######################################################################################################
        # Print in ATCG format just for fun.
        if SHOW_ATCG:
            print()
            for x in gens:
                print(f'{x}:', end='\n')
                
                if type(gens[x]) != dict:
                    if len(gens[x].shape) == 1:
                        print('    ' + ''.join([ATCG_MAPPING[j] for j in gens[x]]))
                    else:
                        for i in range(gens[x].shape[0]):
                            print('    ' + ''.join([ATCG_MAPPING[j] for j in gens[x][i, :]]))
                else:
                    for y in gens[x]:
                        print(f'{y}:', end='\n')
                        if len(gens[x][y].shape) == 1:
                            print('    ' + ''.join([ATCG_MAPPING[j] for j in gens[x][y]]))
                        else:
                            for i in range(gens[x][y].shape[0]):
                                print('    ' + ''.join([ATCG_MAPPING[j] for j in gens[x][y][i, :]]))
                        
        
        if SHOW or SHOW_ATCG:
            print(100 * '=')
            
        print('Gerou os genes com sucesso!')        

    return gens, gens_num
    
def offspring(npc_id1, npc_id2):
    print("Entrou no offspring")
    success = False
    trial = 0

    # Mutação pontual baixa. Como o gene de connections pode ser muito grande,
    # ele usa uma taxa menor.
    MUTATION_RATE = 0.002
    CONNECTION_MUTATION_RATE = 0.0002

    # Probabilidade de preservar exatamente as dimensões de um dos pais.
    # A parte restante mantém a regra anterior de sortear dimensões intermediárias.
    DIM_PARENT_1_PROB = 0.45
    DIM_PARENT_2_PROB = 0.45

    def mutate_array(arr, mutation_rate=MUTATION_RATE):
        arr = np.array(arr, copy=True)

        if mutation_rate <= 0:
            return arr

        mask = np.random.rand(*arr.shape) < mutation_rate

        if np.any(mask):
            arr[mask] = np.random.randint(NUM_SYMBOLS, size=int(mask.sum()))

        return arr

    while not success:
        trial += 1

        if trial == 3 or npc_id1 not in NPC or npc_id2 not in NPC:
            print('Não gerou a child')
            return '', ''

        gens = {}
        gens_num = {}

        ######################################################################################################
        # Dimensions.
        # Regra mais conservadora:
        # - normalmente herda exatamente as dimensões de um dos pais;
        # - às vezes sorteia dimensões intermediárias, como antes.
        u = np.random.rand()

        if u < DIM_PARENT_1_PROB:
            a, b, c = NPC[npc_id1]['dims']
        elif u < DIM_PARENT_1_PROB + DIM_PARENT_2_PROB:
            a, b, c = NPC[npc_id2]['dims']
        else:
            a = np.random.randint(min(NPC[npc_id1]['dims'][0], NPC[npc_id2]['dims'][0]), 1 + max(NPC[npc_id1]['dims'][0], NPC[npc_id2]['dims'][0]))
            b = np.random.randint(min(NPC[npc_id1]['dims'][1], NPC[npc_id2]['dims'][1]), 1 + max(NPC[npc_id1]['dims'][1], NPC[npc_id2]['dims'][1]))
            c = np.random.randint(min(NPC[npc_id1]['dims'][2], NPC[npc_id2]['dims'][2]), 1 + max(NPC[npc_id1]['dims'][2], NPC[npc_id2]['dims'][2]))

        gens['dims'] = [a, b, c]
        gens_num['dims'] = [a, b, c]
        num_neurons = a * b * c

        if SHOW:
            print(f'Child: Dimensions = {a} x {b} x {c}    Number of neurons = {num_neurons}')

        ######################################################################################################
        # Máscara espacial 3D para herança local.
        # A ideia é preservar pacotes coerentes de neurônios:
        # regions[idx], connections[idx], probs[idx] e thr[idx] tendem a vir do mesmo pai.
        axis = np.random.randint(3)
        axis_size = [a, b, c][axis]
        axis_cut = np.random.randint(1, axis_size) if axis_size > 1 else 0
        invert_axis = np.random.rand() > 0.5

        def use_parent1_local(idx):
            x = idx % a
            y = (idx // a) % b
            z = idx // (a * b)
            coord = [x, y, z][axis]

            if axis_size <= 1:
                return np.random.rand() > 0.5

            if invert_axis:
                return coord >= axis_cut
            else:
                return coord < axis_cut

        def parent_order_for_neuron(idx):
            if use_parent1_local(idx):
                return npc_id1, npc_id2
            else:
                return npc_id2, npc_id1

        ######################################################################################################
        # Connections with inputs.
        cut = np.random.randint(int(0.4 * NUM_INPUTS), int(0.6 * NUM_INPUTS)+1)
        gens['inputs'] = np.zeros([NUM_INPUTS, 3 * NUM_DIGITS], dtype=np.int64)

        for i in range(cut):
            gens['inputs'][i, :] = NPC[npc_id1]['gens']['inputs'][i, :]

        for i in range(cut, NUM_INPUTS):
            gens['inputs'][i, :] = NPC[npc_id2]['gens']['inputs'][i, :]

        gens['inputs'] = mutate_array(gens['inputs'])

        gens_num['inputs'] = []

        for n in range(NUM_INPUTS):
            i = digits2int(gens['inputs'][n, : NUM_DIGITS]) % a
            j = digits2int(gens['inputs'][n, NUM_DIGITS : 2*NUM_DIGITS]) % b
            k = digits2int(gens['inputs'][n, 2*NUM_DIGITS : 3*NUM_DIGITS]) % c
            gens_num['inputs'].append(i + j*a + k*a*b)

        gens_num['inputs'] = list(set(gens_num['inputs']))
        num_connections_input_neuron = len(gens_num['inputs'])

        if num_connections_input_neuron != NUM_INPUTS:
            if SHOW:
                print('Child: É necessário repetir a operação pois existem inputs não conectados a neurônios.')
            continue

        if SHOW:
            print(f"Child: Neurons connected to inputs: {gens_num['inputs']}    Number of connections: {num_connections_input_neuron}")

        ######################################################################################################
        # Connections with outputs.
        cut = np.random.randint(int(0.4 * NUM_OUTPUTS), int(0.6 * NUM_OUTPUTS)+1)
        gens['outputs'] = np.zeros([NUM_OUTPUTS, 3 * NUM_DIGITS], dtype=np.int64)

        for i in range(cut):
            gens['outputs'][i, :] = NPC[npc_id1]['gens']['outputs'][i, :]

        for i in range(cut, NUM_OUTPUTS):
            gens['outputs'][i, :] = NPC[npc_id2]['gens']['outputs'][i, :]

        gens['outputs'] = mutate_array(gens['outputs'])

        gens_num['outputs'] = []

        for n in range(NUM_OUTPUTS):
            i = digits2int(gens['outputs'][n, : NUM_DIGITS]) % a
            j = digits2int(gens['outputs'][n, NUM_DIGITS : 2*NUM_DIGITS]) % b
            k = digits2int(gens['outputs'][n, 2*NUM_DIGITS : 3*NUM_DIGITS]) % c
            gens_num['outputs'].append(i + j*a + k*a*b)

        gens_num['outputs'] = list(set(gens_num['outputs']))
        num_connections_output_neuron = len(gens_num['outputs'])

        if num_connections_output_neuron != NUM_OUTPUTS:
            if SHOW:
                print('Child: É necessário repetir a operação pois existem outputs não conectados a neurônios.')
            continue

        success = True

        for x in gens_num['outputs']:
            if x in gens_num['inputs']:
                if SHOW:
                    print(f'Child: É necessário repetir a operação pois um neurônio conectado ao output já está sendo usado por um de input: {x}')
                success = False
                break

        if not success:
            continue

        if SHOW:
            print(f"Child: Neurons connected to outputs: {gens_num['outputs']}    Number of connections: {num_connections_output_neuron}")

        ######################################################################################################
        # Output thresholds to make action.
        cut = np.random.randint(int(0.4 * len(gens['outputs'])), int(0.6 * len(gens['outputs']))+1)
        gens['outputs_thr'] = np.zeros([len(gens['outputs'])], dtype=np.int64)

        for i in range(cut):
            gens['outputs_thr'][i] = NPC[npc_id1]['gens']['outputs_thr'][i]

        for i in range(cut, len(gens['outputs'])):
            gens['outputs_thr'][i] = NPC[npc_id2]['gens']['outputs_thr'][i]

        gens['outputs_thr'] = mutate_array(gens['outputs_thr'])

        gens_num['outputs_thr'] = {}
        w = 0

        for x in gens_num['outputs']:
            gens_num['outputs_thr'][x] = MAPPING[gens['outputs_thr'][w]]
            w += 1

        if SHOW:
            print(f"Child: Output threshold of neuron {x} = {gens_num['outputs_thr'][x]}, ...")

        ######################################################################################################
        # Region sizes.
        # Agora regions são herdadas por pacote local espacial, não por corte independente.
        gens['regions'] = np.zeros([num_neurons, 3 * NUM_DIGITS_2], dtype=np.int64)

        for idx in range(num_neurons):
            parent_first, parent_second = parent_order_for_neuron(idx)

            if idx < NPC[parent_first]['gens']['regions'].shape[0]:
                gens['regions'][idx, :] = NPC[parent_first]['gens']['regions'][idx, :]
            elif idx < NPC[parent_second]['gens']['regions'].shape[0]:
                gens['regions'][idx, :] = NPC[parent_second]['gens']['regions'][idx, :]
            else:
                gens['regions'][idx, :] = np.random.randint(NUM_SYMBOLS, size=[3 * NUM_DIGITS_2])

        gens['regions'] = mutate_array(gens['regions'])

        gens_num['regions'] = {}
        w = 0

        for i in range(a):
            for j in range(b):
                for k in range(c):
                    region_size_x = digits2int(gens['regions'][w, : NUM_DIGITS_2]) % a
                    region_size_y = digits2int(gens['regions'][w, NUM_DIGITS_2 : 2*NUM_DIGITS_2]) % b
                    region_size_z = digits2int(gens['regions'][w, 2*NUM_DIGITS_2 : 3*NUM_DIGITS_2]) % c
                    gens_num['regions'][i + j*a + k*a*b] = [region_size_x, region_size_y, region_size_z]
                    w += 1

        ######################################################################################################
        # Conexões pré-programadas no cérebro. Cada neurônio está conectado a um conjunto de neurônios diferentes no início.
        if np.random.rand() > 0.5:
            gens['num_initial_connections'] = np.array(NPC[npc_id1]['gens']['num_initial_connections'], copy=True)
        else:
            gens['num_initial_connections'] = np.array(NPC[npc_id2]['gens']['num_initial_connections'], copy=True)

        gens['num_initial_connections'] = mutate_array(gens['num_initial_connections'])

        num_initial_connections = digits2int(gens['num_initial_connections']) % (num_neurons - 1)
        gens_num['num_initial_connections'] = num_initial_connections
        gens['connections'] = {}
        gens_num['connections'] = {}

        for i in range(a):
            for j in range(b):
                for k in range(c):
                    idx = i + j*a + k*a*b
                    parent_first, parent_second = parent_order_for_neuron(idx)

                    gens['connections'][idx] = np.zeros([num_initial_connections, 3 * NUM_DIGITS_2], dtype=np.int64)

                    for u in range(num_initial_connections):
                        if idx in NPC[parent_first]['gens']['connections'] and u < NPC[parent_first]['gens']['connections'][idx].shape[0]:
                            gens['connections'][idx][u, :] = NPC[parent_first]['gens']['connections'][idx][u, :]
                        elif idx in NPC[parent_second]['gens']['connections'] and u < NPC[parent_second]['gens']['connections'][idx].shape[0]:
                            gens['connections'][idx][u, :] = NPC[parent_second]['gens']['connections'][idx][u, :]
                        else:
                            gens['connections'][idx][u, :] = np.random.randint(NUM_SYMBOLS, size=[3 * NUM_DIGITS_2])

                    gens['connections'][idx] = mutate_array(gens['connections'][idx], CONNECTION_MUTATION_RATE)

                    gens_num['connections'][idx] = []

                    for w in range(num_initial_connections):
                        neuron_x = digits2int(gens['connections'][idx][w, : NUM_DIGITS_2]) % a
                        neuron_y = digits2int(gens['connections'][idx][w, NUM_DIGITS_2 : 2*NUM_DIGITS_2]) % b
                        neuron_z = digits2int(gens['connections'][idx][w, 2*NUM_DIGITS_2 : 3*NUM_DIGITS_2]) % c
                        neuron2 = neuron_x + neuron_y*a + neuron_z*a*b

                        # Verifica se não é conexão repetida.
                        if neuron2 not in gens_num['connections'][idx]:
                            # Verifica se já não há uma conexão na direção contrária.
                            if (neuron2 not in gens_num['connections']) or (idx not in gens_num['connections'][neuron2]):
                                # Verifica se o neurônio candidato está dentro da região permitida.
                                radius_x, radius_y, radius_z = gens_num['regions'][idx]
                                if i - radius_x <= neuron_x <= i + radius_x:
                                    if j - radius_y <= neuron_y <= j + radius_y:
                                        if k - radius_z <= neuron_z <= k + radius_z:
                                            gens_num['connections'][idx].append(neuron2)

        if SHOW:
            print(f"Child: Number of initial connections per neuron = {gens_num['num_initial_connections']}")

        ######################################################################################################
        # Probabilities of making connections.
        # Agora probs seguem o mesmo pacote local usado em regions/connections.
        gens['probs'] = np.zeros([num_neurons], dtype=np.int64)

        for idx in range(num_neurons):
            parent_first, parent_second = parent_order_for_neuron(idx)

            if idx < NPC[parent_first]['gens']['probs'].shape[0]:
                gens['probs'][idx] = NPC[parent_first]['gens']['probs'][idx]
            elif idx < NPC[parent_second]['gens']['probs'].shape[0]:
                gens['probs'][idx] = NPC[parent_second]['gens']['probs'][idx]
            else:
                gens['probs'][idx] = np.random.randint(NUM_SYMBOLS)

        gens['probs'] = mutate_array(gens['probs'])

        gens_num['probs'] = {}
        w = 0

        for i in range(a):
            for j in range(b):
                for k in range(c):
                    gens_num['probs'][i + j*a + k*a*b] = MAPPING[gens['probs'][w]]
                    w += 1

        ######################################################################################################
        # Threshold per neuron.
        # Agora thr também segue o mesmo pacote local.
        gens['thr'] = np.zeros([num_neurons], dtype=np.int64)

        for idx in range(num_neurons):
            parent_first, parent_second = parent_order_for_neuron(idx)

            if idx < NPC[parent_first]['gens']['thr'].shape[0]:
                gens['thr'][idx] = NPC[parent_first]['gens']['thr'][idx]
            elif idx < NPC[parent_second]['gens']['thr'].shape[0]:
                gens['thr'][idx] = NPC[parent_second]['gens']['thr'][idx]
            else:
                gens['thr'][idx] = np.random.randint(NUM_SYMBOLS)

        gens['thr'] = mutate_array(gens['thr'])

        gens_num['thr'] = {}
        w = 0

        for i in range(a):
            for j in range(b):
                for k in range(c):
                    gens_num['thr'][i + j*a + k*a*b] = MAPPING[gens['thr'][w]]
                    w += 1

        ######################################################################################################
        # Decay of neuron connections.
        if np.random.rand() > 0.5:
            gens['decay'] = np.array(NPC[npc_id1]['gens']['decay'], copy=True)
        else:
            gens['decay'] = np.array(NPC[npc_id2]['gens']['decay'], copy=True)

        gens['decay'] = mutate_array(gens['decay'])
        gens_num['decay'] = float('0.' + ''.join([str(x) for x in gens['decay']]))

        if SHOW:
            print(f"Child: Decay e = {gens_num['decay']}")

        # Number of iterations that the neuron is inactive after sending a signal.
        # Antes este gene era sempre aleatório. Agora ele é herdado com baixa chance de mutação,
        # para preservar melhor filhos de bons pais.
        if np.random.rand() > 0.5:
            gens['inactive_iter'] = np.array(NPC[npc_id1]['gens']['inactive_iter'], copy=True)
        else:
            gens['inactive_iter'] = np.array(NPC[npc_id2]['gens']['inactive_iter'], copy=True)

        gens['inactive_iter'] = mutate_array(gens['inactive_iter'])
        gens_num['inactive_iter'] = digits2int(gens['inactive_iter']) + 1

        if SHOW:
            print(f"Child: Number of inactive iterations after sending a signal = {gens_num['inactive_iter']}")

        # Exponential factor saturation response of the signals.
        if np.random.rand() > 0.5:
            gens['exp_response'] = np.array(NPC[npc_id1]['gens']['exp_response'], copy=True)
        else:
            gens['exp_response'] = np.array(NPC[npc_id2]['gens']['exp_response'], copy=True)

        gens['exp_response'] = mutate_array(gens['exp_response'])
        gens_num['exp_response'] = digits2int(gens['exp_response'])//10 + 1

        if SHOW:
            print(f"Child: Exponential factor saturation response of the signals = {gens_num['exp_response']}")

        # Number of iterations that the neuron is weakened after sending a signal.
        if np.random.rand() > 0.5:
            gens['weakened_iter'] = np.array(NPC[npc_id1]['gens']['weakened_iter'], copy=True)
        else:
            gens['weakened_iter'] = np.array(NPC[npc_id2]['gens']['weakened_iter'], copy=True)

        gens['weakened_iter'] = mutate_array(gens['weakened_iter'])
        gens_num['weakened_iter'] = gens_num['inactive_iter'] + digits2int(gens['weakened_iter']) + 1

        if SHOW:
            print(f"Child: Number of weakened iterations after sending a signal = {gens_num['weakened_iter']}")

        ######################################################################################################
        # Print in ATCG format just for fun.
        if SHOW_ATCG:
            print()
            for x in gens:
                print(f'Child: {x}:', end='\n')

                if type(gens[x]) != dict:
                    if hasattr(gens[x], 'shape') and len(gens[x].shape) == 1:
                        print('    ' + ''.join([ATCG_MAPPING[j] for j in gens[x]]))
                    elif hasattr(gens[x], 'shape'):
                        for i in range(gens[x].shape[0]):
                            print('    ' + ''.join([ATCG_MAPPING[j] for j in gens[x][i, :]]))
                    else:
                        print(f'    {gens[x]}')
                else:
                    for y in gens[x]:
                        print(f'Child: {y}:', end='\n')
                        if len(gens[x][y].shape) == 1:
                            print('    ' + ''.join([ATCG_MAPPING[j] for j in gens[x][y]]))
                        else:
                            for i in range(gens[x][y].shape[0]):
                                print('    ' + ''.join([ATCG_MAPPING[j] for j in gens[x][y][i, :]]))

        if SHOW or SHOW_ATCG:
            print(100 * '=')

        print('Child: Gerou os genes com sucesso!')

    return gens, gens_num   
    
def extras(npc_id):
    npc = NPC[npc_id]

    npc['neuron_signals'] = {}
    npc['connection_strengths'] = {}
    npc['inactive_counter'] = {
        neuron: npc['inactive_iter']
        for neuron in npc['connections']
    }

    # Mantém o mesmo significado de antes, mas sem recalcular toda hora.
    npc['num_connections'] = sum(len(conns) for conns in npc['connections'].values())
    npc['num_connections_count'] = npc['num_connections']

    # Estrutura auxiliar só para acelerar:
    # - "neuron2 in my_connections"
    # - remoções
    # A lista original continua sendo a fonte de verdade para a ordem.
    npc['connections_set'] = {neuron: set(conns) for neuron, conns in npc['connections'].items()}

    npc['num_childs'] = 0

    for neuron, conns in npc['connections'].items():
        npc['connection_strengths'][neuron] = {neuron2: 1 for neuron2 in conns}

    return

def digits2int(digits):
    return int(''.join([str(x) for x in digits]))
    
def exp_response(x, k):
    return 1.0 - np.exp(-x/k)

# Função simulando a lógica de decisão de um NPC
def process_npc(npc_id, inputs):
    if npc_id in NPC:
        neuron_signals = NPC[npc_id]['neuron_signals']
        neuron_thrs = NPC[npc_id]['thr']
        neuron_connections = NPC[npc_id]['connections']
        e_orig = NPC[npc_id]['decay']
        e_now = NPC[npc_id]['connection_strengths']
        exp_factor = NPC[npc_id]['exp_response']
            
        if npc_id == 'blob_0':
            neuron_signals_tmp, to_zero = process_npc_neurons_blob_0(npc_id, neuron_signals, neuron_thrs, neuron_connections, e_orig, e_now, exp_factor)              
        else:
            neuron_signals_tmp, to_zero = process_npc_neurons(npc_id, neuron_signals, neuron_thrs, neuron_connections, e_orig, e_now, exp_factor)
            
        process_npc_neurons_updates(npc_id, neuron_signals, neuron_signals_tmp, to_zero)
        process_npc_inputs(npc_id, inputs)
        outputs = process_npc_outputs(npc_id)  
    else:
        outputs = None
            
    return outputs
    
def process_npc_neurons_blob_0(npc_id, neuron_signals, neuron_thrs, neuron_connections, e_orig, e_now, exp_factor):
    global it
    global it_real
    global start

    it_real += 1

    if it_real == 0:
        start = time.time()
        fps_real = ''
    else:
        total_time = time.time() - start
        fps_real = int(it_real / total_time)

    if it_real % N == 0 or it_real < N:
        it += 1

    print(f'Processando sinais - iteração {it_real} ({fps_real} it/s)')

    npc = NPC[npc_id]
    neuron_inactive_counter = npc['inactive_counter']
    neuron_inactive_iter = npc['inactive_iter']

    neuron_signals_tmp = {}
    to_zero = []

    exp_response_local = exp_response
    update_strengthening = update_connections_strengthening
    create_conn = create_new_connections
    update_weakening = update_connections_weakening

    for neuron, signal in neuron_signals.items():
        thr = neuron_thrs[neuron]

        blob_history.append(f'{it} {neuron} {signal}\n')

        if signal > thr and neuron_inactive_counter[neuron] == neuron_inactive_iter:
            neuron_inactive_counter[neuron] -= 1

            signal = exp_response_local(signal, exp_factor)
            to_zero.append(neuron)

            conn_list = neuron_connections[neuron]
            conn_strengths = e_now[neuron]

            for neuron2 in conn_list:
                e = (1 + conn_strengths[neuron2]) / 2
                update_strengthening(npc_id, e, neuron, neuron2)

                if neuron2 in neuron_signals_tmp:
                    neuron_signals_tmp[neuron2] += e * signal
                else:
                    neuron_signals_tmp[neuron2] = e * signal

            create_conn(npc_id, neuron)

        if neuron_inactive_counter[neuron] != neuron_inactive_iter:
            neuron_inactive_counter[neuron] -= 1
            if neuron_inactive_counter[neuron] < 0:
                neuron_inactive_counter[neuron] = neuron_inactive_iter

        update_weakening(npc_id, e_orig, e_now, neuron)

    return neuron_signals_tmp, to_zero
    
def process_npc_neurons(npc_id, neuron_signals, neuron_thrs, neuron_connections, e_orig, e_now, exp_factor):
    npc = NPC[npc_id]
    neuron_inactive_counter = npc['inactive_counter']
    neuron_inactive_iter = npc['inactive_iter']
    
    # Propaga os sinais de neurônios para neurônios.
    neuron_signals_tmp = {}
    to_zero = []
    
    for neuron, signal in neuron_signals.items():
        thr = neuron_thrs[neuron]        
        
        if signal > thr:
            if neuron_inactive_counter[neuron] == neuron_inactive_iter:
                neuron_inactive_counter[neuron] -= 1
                                    
                signal = exp_response(signal, exp_factor)
                to_zero.append(neuron)
                
                for neuron2 in neuron_connections[neuron]: 
                    e = (1 + e_now[neuron][neuron2]) / 2            
                    update_connections_strengthening(npc_id, e, neuron, neuron2)
                    
                    if neuron2 in neuron_signals_tmp:
                        neuron_signals_tmp[neuron2] += e * signal  
                    else:
                        neuron_signals_tmp[neuron2] = e * signal
                        
                # Depois de repassar os sinais, checa se algum outro neurônio na região também está com sinal e tenta criar uma conexão.
                create_new_connections(npc_id, neuron)
                                                    
        # Período de retração do neurônio. Acaba quando o contador ficar negativo.
        if neuron_inactive_counter[neuron] != neuron_inactive_iter:
            neuron_inactive_counter[neuron] -= 1
            
            if neuron_inactive_counter[neuron] < 0:
                neuron_inactive_counter[neuron] = neuron_inactive_iter

        update_connections_weakening(npc_id, e_orig, e_now, neuron)                
                
    return neuron_signals_tmp, to_zero
    
def create_new_connections(npc_id, neuron):
    npc = NPC[npc_id]

    a, b, c = npc['dims']
    radius_x, radius_y, radius_z = npc['regions'][neuron]

    neuron_x = neuron % a
    neuron_y = (neuron // a) % b
    neuron_z = neuron // (a * b)

    neuron_probs = npc['probs'][neuron]
    neuron_signals = npc['neuron_signals']
    neuron_connections = npc['connections']
    neuron_connections_set = npc['connections_set']
    neuron_connection_strengths = npc['connection_strengths']

    my_connections = neuron_connections[neuron]
    my_connections_set = neuron_connections_set[neuron]
    my_strengths = neuron_connection_strengths[neuron]

    range1 = range(max(0, neuron_x - radius_x), min(a, neuron_x + radius_x))
    range2 = range(max(0, neuron_y - radius_y), min(b, neuron_y + radius_y))
    range3 = range(max(0, neuron_z - radius_z), min(c, neuron_z + radius_z))

    # Mantém exatamente a mesma ideia de amostragem parcial.
    range1 = random.sample(range1, 1 + len(range1) // 2) if len(range1) > 1 else range1
    range2 = random.sample(range2, 1 + len(range2) // 2) if len(range2) > 1 else range2
    range3 = random.sample(range3, 1 + len(range3) // 2) if len(range3) > 1 else range3

    for i in range1:
        for j in range2:
            for k in range3:
                neuron2 = i + j * a + k * a * b

                if neuron_probs < np.random.rand():
                    continue
                if neuron2 not in neuron_signals:
                    continue
                if neuron_signals[neuron2] <= 0:
                    continue
                if neuron2 in my_connections_set or neuron in neuron_connections_set[neuron2]:
                    continue

                my_connections.append(neuron2)
                my_connections_set.add(neuron2)
                my_strengths[neuron2] = 1
                npc['num_connections_count'] += 1
                return

    return
                        
def update_connections_weakening(npc_id, e_orig, e_now, neuron):
    npc = NPC[npc_id]

    neuron_outputs = npc['outputs']
    neuron_inputs = npc['inputs']
    neuron_connections = npc['connections']
    neuron_connection_strengths = npc['connection_strengths']
    neuron_connections_set = npc['connections_set']

    my_connections = neuron_connections[neuron]
    my_connections_set = neuron_connections_set[neuron]
    my_strengths = neuron_connection_strengths[neuron]

    blocked = (neuron in neuron_outputs) or (neuron in neuron_inputs)

    for neuron2 in list(e_now[neuron].keys()):
        if blocked or (neuron2 in neuron_outputs) or (neuron2 in neuron_inputs):
            continue

        e = e_now[neuron][neuron2] - e_orig**2

        if e < 0:
            # remove na lista + set, preservando o comportamento original
            if neuron2 in my_connections_set:
                my_connections_set.remove(neuron2)
                my_connections.remove(neuron2)
                del my_strengths[neuron2]
                npc['num_connections_count'] -= 1
        else:
            my_strengths[neuron2] = e

    return
    
def update_connections_strengthening(npc_id, e, neuron, neuron2):
    NPC[npc_id]['connection_strengths'][neuron][neuron2] = e
        
    return 
    
def update_connections_strengthening(npc_id, e, neuron, neuron2):
    npc = NPC[npc_id]
    npc['connection_strengths'][neuron][neuron2] = e
    return
    
def process_npc_neurons(npc_id, neuron_signals, neuron_thrs, neuron_connections, e_orig, e_now, exp_factor):
    npc = NPC[npc_id]
    neuron_inactive_counter = npc['inactive_counter']
    neuron_inactive_iter = npc['inactive_iter']

    neuron_signals_tmp = {}
    to_zero = []

    exp_response_local = exp_response
    update_strengthening = update_connections_strengthening
    create_conn = create_new_connections
    update_weakening = update_connections_weakening

    for neuron, signal in neuron_signals.items():
        thr = neuron_thrs[neuron]

        if signal > thr and neuron_inactive_counter[neuron] == neuron_inactive_iter:
            neuron_inactive_counter[neuron] -= 1

            signal = exp_response_local(signal, exp_factor)
            to_zero.append(neuron)

            conn_list = neuron_connections[neuron]
            conn_strengths = e_now[neuron]

            for neuron2 in conn_list:
                e = (1 + conn_strengths[neuron2]) / 2
                update_strengthening(npc_id, e, neuron, neuron2)

                if neuron2 in neuron_signals_tmp:
                    neuron_signals_tmp[neuron2] += e * signal
                else:
                    neuron_signals_tmp[neuron2] = e * signal

            create_conn(npc_id, neuron)

        if neuron_inactive_counter[neuron] != neuron_inactive_iter:
            neuron_inactive_counter[neuron] -= 1
            if neuron_inactive_counter[neuron] < 0:
                neuron_inactive_counter[neuron] = neuron_inactive_iter

        update_weakening(npc_id, e_orig, e_now, neuron)

    return neuron_signals_tmp, to_zero
    
def process_npc_inputs(npc_id, inputs):
    npc = NPC[npc_id]
    neuron_intputs = npc['inputs']
    neuron_signals = npc['neuron_signals']

    # Propaga os sinais de inputs para neurônios.
    exp_factor = npc['exp_response']
    
    for neuron, signal in zip(neuron_intputs, inputs):
        neuron_signals[neuron] = exp_response(signal, exp_factor)
        
    return
    
def process_npc_outputs(npc_id):
    npc = NPC[npc_id]
    neuron_outputs = npc['outputs']
    neuron_signals = npc['neuron_signals']
    neuron_outputs_thr = npc['outputs_thr']
    exp_factor = npc['exp_response']

    outputs = [0] * len(neuron_outputs)

    for i, neuron in enumerate(neuron_outputs):
        if neuron in neuron_signals:
            signal = neuron_signals[neuron]
            outputs[i] = 1 if exp_response(signal, exp_factor) > neuron_outputs_thr[neuron] else 0

    return outputs
                
# Inicia o servidor WebSocket local
if __name__ == "__main__":
    # uvicorn.run(app, host="127.0.0.1", port=8000) # Inicia o servidor na porta 8000
    config = uvicorn.Config(app, host="127.0.0.1", port=8000)
    server = uvicorn.Server(config)
    server.run()