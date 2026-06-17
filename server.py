import json
import time
import random
import asyncio
import uvicorn
import datetime
import traceback
import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI, WebSocket

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
SHOW = False
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
        counter = 0
        mod = 5
        
        while True:
            try:
                data = await websocket.receive_text()  # Recebe mensagem do client
            except:
                break
                
            payload = json.loads(data)  # Converte a mensagem para JSON
            npcs = payload.get("batch", [])  # Obtém os dados dos NPCs
            results = []  # Lista para armazenar os resultados     
 
            for npc in npcs:
                npc_action = npc.get("npc_action")    # process ou gen_brain
                npc_id = npc.get("id")  # Obtém o ID do NPC
                inputs = npc.get("inputs")  # Obtém os inputs do NPC
                                
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
                elif npc_action == "process":
                    # ECONOMIA: faz apenas alguns dos blobs por request
                    if int_id != 0:
                        if int_id % mod != counter:
                            continue                 
                
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
                        results.append({"id": gens_num["id"], 
                                        "npc_action": "offspring"}) 
                        NPC[npc_id] = gens_num  
                        NPC[npc_id]['gens'] = gens  # Memoriza o código genético puro
                        extras(npc_id)  
                   
            response = json.dumps({"results": results})  # Cria a resposta com os resultados
            await websocket.send_text(response)  # Envia a resposta ao cliente
            counter += 1
            counter = counter % mod
            
            # Remove do NPC global as ids mortas.
            current_alive = {npc.get("id") for npc in npcs}
            
            if len(current_alive) > 1 and npc_action == "process":
                for npc_id in list(NPC.keys()):
                    if npc_id != "blob_0" and npc_id not in current_alive:
                        del NPC[npc_id]
            
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

        await websocket.close()
        print("🔒 Conexão WebSocket encerrada.")
        server.should_exit = True

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
    success = False
    trial = 0
        
    
    while not success:
        trial += 1
        
        if trial == 3 or npc_id1 not in NPC or npc_id2 not in NPC:
            return '', ''
        
        gens = {}
        gens_num = {}

        ######################################################################################################
        # Dimensions.
        a = np.random.randint(min(NPC[npc_id1]['dims'][0], NPC[npc_id2]['dims'][0]), 1 + max(NPC[npc_id1]['dims'][0], NPC[npc_id2]['dims'][0]))
        b = np.random.randint(min(NPC[npc_id1]['dims'][1], NPC[npc_id2]['dims'][1]), 1 + max(NPC[npc_id1]['dims'][1], NPC[npc_id2]['dims'][1]))
        c = np.random.randint(min(NPC[npc_id1]['dims'][2], NPC[npc_id2]['dims'][2]), 1 + max(NPC[npc_id1]['dims'][2], NPC[npc_id2]['dims'][2]))
        gens['dims'] = [a, b, c]
        gens_num['dims'] = [a, b, c]
        num_neurons = a * b * c
        
        if SHOW:
            print(f'Child: Dimensions = {a} x {b} x {c}    Number of neurons = {num_neurons}')

        ######################################################################################################
        # Connections with inputs.
        cut = np.random.randint(int(0.4 * NUM_INPUTS), int(0.6 * NUM_INPUTS)+1)
        gens['inputs'] = np.zeros([NUM_INPUTS, 3 * NUM_DIGITS], dtype=np.int64)
        
        for i in range(cut):
            gens['inputs'][i, :] = NPC[npc_id1]['gens']['inputs'][i, :]
            
        for i in range(cut, NUM_INPUTS):
            gens['inputs'][i, :] = NPC[npc_id2]['gens']['inputs'][i, :]
                    
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

        for x in gens_num['outputs']:
            if x in gens_num['inputs']:
                if SHOW:
                    print(f'Child: É necessário repetir a operação pois um neurônio conectado ao output já está sendo usado por um de input: {x}')
                success = False
                break
            success = True
            
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
        
        gens_num['outputs_thr'] = {}
        w = 0

        for x in gens_num['outputs']:
            gens_num['outputs_thr'][x] = MAPPING[gens['outputs_thr'][w]]
            w += 1

        if SHOW:
            print(f"Child: Output threshold of neuron {x} = {gens_num['outputs_thr'][x]}, ...")

        ######################################################################################################
        # Region sizes.
        cut = np.random.randint(int(0.4 * num_neurons), int(0.6 * num_neurons)+1)
        gens['regions'] = np.zeros([num_neurons, 3 * NUM_DIGITS_2], dtype=np.int64)
        
        for i in range(cut):
            if i < NPC[npc_id1]['gens']['regions'].shape[0]:
                gens['regions'][i, :] = NPC[npc_id1]['gens']['regions'][i, :]
            elif i < NPC[npc_id2]['gens']['regions'].shape[0]:
                gens['regions'][i, :] = NPC[npc_id2]['gens']['regions'][i, :]
            else:    # fallback - filho tem mais neurônios que os pais
                gens['regions'][i, :] = np.random.randint(NUM_SYMBOLS, size=[3 * NUM_DIGITS_2])
            
        for i in range(cut, num_neurons):
           if i < NPC[npc_id2]['gens']['regions'].shape[0]:
               gens['regions'][i, :] = NPC[npc_id2]['gens']['regions'][i, :]
           elif i < NPC[npc_id1]['gens']['regions'].shape[0]:
                gens['regions'][i, :] = NPC[npc_id1]['gens']['regions'][i, :]
           else:    # fallback - filho tem mais neurônios que os pais
                gens['regions'][i, :] = np.random.randint(NUM_SYMBOLS, size=[3 * NUM_DIGITS_2])
        
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
        
        if np.random.rand() > 0.5:
            gens['num_initial_connections'] = NPC[npc_id1]['gens']['num_initial_connections']
        else:
            gens['num_initial_connections'] = NPC[npc_id2]['gens']['num_initial_connections']
        
        num_initial_connections = digits2int(gens['num_initial_connections']) % (num_neurons - 1)
        gens_num['num_initial_connections'] = num_initial_connections
        gens['connections'] = {}
        gens_num['connections'] = {}

        for i in range(a):
            for j in range(b):
                for k in range(c):
                    idx = i + j*a + k*a*b
                    cut = np.random.randint(int(0.4 * num_initial_connections), int(0.6 * num_initial_connections)+1)
                    gens['connections'][idx] = np.zeros([num_initial_connections, 3 * NUM_DIGITS_2], dtype=np.int64)
                    
                    for u in range(cut):
                        if idx in NPC[npc_id1]['gens']['connections'] and u < NPC[npc_id1]['gens']['connections'][idx].shape[0]:
                            gens['connections'][idx][u, :] = NPC[npc_id1]['gens']['connections'][idx][u, :]
                        elif idx in NPC[npc_id2]['gens']['connections'] and u < NPC[npc_id2]['gens']['connections'][idx].shape[0]:
                            gens['connections'][idx][u, :] = NPC[npc_id2]['gens']['connections'][idx][u, :]
                        else:    # fallback - filho tem mais neurônios que os pais
                            gens['connections'][idx][u, :] = np.random.randint(NUM_SYMBOLS, size=[3 * NUM_DIGITS_2])
                                                
                    for u in range(cut, num_initial_connections):
                        if idx in NPC[npc_id2]['gens']['connections'] and u < NPC[npc_id2]['gens']['connections'][idx].shape[0]:
                            gens['connections'][idx][u, :] = NPC[npc_id2]['gens']['connections'][idx][u, :]
                        elif idx in NPC[npc_id1]['gens']['connections'] and u < NPC[npc_id1]['gens']['connections'][idx].shape[0]:
                            gens['connections'][idx][u, :] = NPC[npc_id1]['gens']['connections'][idx][u, :]
                        else:    # fallback - filho tem mais neurônios que os pais
                            gens['connections'][idx][u, :] = np.random.randint(NUM_SYMBOLS, size=[3 * NUM_DIGITS_2])
                                            
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
        cut = np.random.randint(int(0.4 * num_neurons), int(0.6 * num_neurons)+1)
        gens['probs'] = np.zeros([num_neurons], dtype=np.int64)
        
        for i in range(cut):
            if i < NPC[npc_id1]['gens']['probs'].shape[0]:
                gens['probs'][i] = NPC[npc_id1]['gens']['probs'][i]
            elif i < NPC[npc_id2]['gens']['probs'].shape[0]:
                gens['probs'][i] = NPC[npc_id2]['gens']['probs'][i]
            else:    # fallback - filho tem mais neurônios que os pais
                gens['probs'][i] = np.random.randint(NUM_SYMBOLS)
            
        for i in range(cut, num_neurons):
            if i < NPC[npc_id2]['gens']['probs'].shape[0]:
                gens['probs'][i] = NPC[npc_id2]['gens']['probs'][i]
            elif i < NPC[npc_id1]['gens']['probs'].shape[0]:
                gens['probs'][i] = NPC[npc_id1]['gens']['probs'][i]
            else:    # fallback - filho tem mais neurônios que os pais
                gens['probs'][i] = np.random.randint(NUM_SYMBOLS)
        
        gens_num['probs'] = {}
        w = 0

        for i in range(a):
            for j in range(b):
                for k in range(c):
                    gens_num['probs'][i + j*a + k*a*b] = MAPPING[gens['probs'][w]]
                    w += 1

        ######################################################################################################
        # Threshold per neuron.
        cut = np.random.randint(int(0.4 * num_neurons), int(0.6 * num_neurons)+1)
        gens['thr'] = np.zeros([num_neurons], dtype=np.int64)
        
        for i in range(cut):
            if i < NPC[npc_id1]['gens']['thr'].shape[0]:
                gens['thr'][i] = NPC[npc_id1]['gens']['thr'][i]
            elif i < NPC[npc_id2]['gens']['thr'].shape[0]:
                gens['thr'][i] = NPC[npc_id2]['gens']['thr'][i]
            else:    # fallback - filho tem mais neurônios que os pais
                gens['thr'][i] = np.random.randint(NUM_SYMBOLS)
            
        for i in range(cut, num_neurons):
            if i < NPC[npc_id2]['gens']['thr'].shape[0]:
                gens['thr'][i] = NPC[npc_id2]['gens']['thr'][i]
            elif i < NPC[npc_id1]['gens']['thr'].shape[0]:
                gens['thr'][i] = NPC[npc_id1]['gens']['thr'][i]
            else:    # fallback - filho tem mais neurônios que os pais
                gens['thr'][i] = np.random.randint(NUM_SYMBOLS)
        
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
            gens['decay'] = NPC[npc_id1]['gens']['decay']
        else:
            gens['decay'] = NPC[npc_id2]['gens']['decay']
        
        gens_num['decay'] = float('0.' + ''.join([str(x) for x in gens['decay']]))
        
        if SHOW:
            print(f"Child: Decay e = {gens_num['decay']}")

        # Number of iterations that the neuron is inactive after sending a signal.
        gens['inactive_iter'] = np.random.randint(NUM_SYMBOLS, size=[NUM_DIGITS_2])
        gens_num['inactive_iter'] = digits2int(gens['inactive_iter']) + 1
        
        if SHOW:
            print(f"Child: Number of inactive iterations after sending a signal = {gens_num['inactive_iter']}")
            
        # Exponential factor saturation response of the signals.
        if np.random.rand() > 0.5:
            gens['exp_response'] = NPC[npc_id1]['gens']['exp_response']
        else:
            gens['exp_response'] = NPC[npc_id2]['gens']['exp_response']
        
        gens_num['exp_response'] = digits2int(gens['exp_response'])//10 + 1
        
        if SHOW:
            print(f"Child: Exponential factor saturation response of the signals = {gens_num['exp_response']}")

        # Number of iterations that the neuron is weakened after sending a signal.
        if np.random.rand() > 0.5:
            gens['weakened_iter'] = NPC[npc_id1]['gens']['weakened_iter']
        else:
            gens['weakened_iter'] = NPC[npc_id2]['gens']['weakened_iter']
        
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
                    if len(gens[x].shape) == 1:
                        print('    ' + ''.join([ATCG_MAPPING[j] for j in gens[x]]))
                    else:
                        for i in range(gens[x].shape[0]):
                            print('    ' + ''.join([ATCG_MAPPING[j] for j in gens[x][i, :]]))
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
    NPC[npc_id]['neuron_signals'] = {}
    NPC[npc_id]['connection_strengths'] = {}
    NPC[npc_id]['inactive_counter'] = {neuron: NPC[npc_id]['inactive_iter'] for neuron in NPC[npc_id]['connections']}
    NPC[npc_id]['num_connections'] = sum([len(NPC[npc_id]['connections'][neuron]) for neuron in NPC[npc_id]['connections']])
    NPC[npc_id]['num_childs'] = 0
    
    for neuron in NPC[npc_id]['connections']:
        NPC[npc_id]['connection_strengths'][neuron] = {}
        for neuron2 in NPC[npc_id]['connections'][neuron]:
            NPC[npc_id]['connection_strengths'][neuron][neuron2] = 1
                
    return

def digits2int(digits):
    return int(''.join([str(x) for x in digits]))
    
def exp_response(x, k):
    return 1.0 - np.exp(-x/k)

# Função simulando a lógica de decisão de um NPC
def process_npc(npc_id, inputs):
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
        end_frame = time.time()
        total_time = end_frame - start
        fps_real = int(it_real / total_time)
    
    if it_real % N == 0 or it_real < N: 
        it += 1        
        
    print(f'Processando sinais - iteração {it_real} ({fps_real} it/s)')
    
    npc = NPC[npc_id]
    neuron_inactive_counter = npc['inactive_counter']
    neuron_inactive_iter = npc['inactive_iter']
    
    # Propaga os sinais de neurônios para neurônios.
    neuron_signals_tmp = {}
    to_zero = []
    
    for neuron, signal in neuron_signals.items():
        thr = neuron_thrs[neuron]        
        
        blob_history.append(f'{it} {neuron} {signal}\n')
        
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
                
        # Atualiza o decaimento da força das conexões do neurônio.
        update_connections_weakening(npc_id, e_orig, e_now, neuron)
                
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
    #neuron_x, neuron_y, neuron_z = [int(x) for x in neuron.split('-')]
    neuron_x, neuron_y, neuron_z = neuron % a, (neuron // a) % b, neuron // (a * b)
    neuron_probs = npc['probs'][neuron]
    neuron_signals = npc['neuron_signals']
    neuron_connections = npc['connections']
    neuron_connection_strengths = npc['connection_strengths']
    my_connections = neuron_connections[neuron]
    my_strengths = neuron_connection_strengths[neuron]
    range1 = range(max(0, neuron_x - radius_x), min(a, neuron_x + radius_x)) 
    range2 = range(max(0, neuron_y - radius_y), min(b, neuron_y + radius_y)) 
    range3 = range(max(0, neuron_z - radius_z), min(c, neuron_z + radius_z)) 
    
    # ECONOMIA: Percorre apenas alguns neurõnios da região para tentar criar novas conexões.
    range1 = random.sample(range1, 1 + len(range1) // 2) if len(range1) > 1 else range1
    range2 = random.sample(range2, 1 + len(range2) // 2) if len(range2) > 1 else range2
    range3 = random.sample(range3, 1 + len(range3) // 2) if len(range3) > 1 else range3
    candidates = [i + j*a + k*a*b for i in range1 for j in range2 for k in range3]    
    
    for neuron2 in candidates:
        # Antes de começar qualquer coisa, faz um teste de probabilidade.
        if neuron_probs < np.random.rand():
            continue
                
        # Verifica se o neurônio está no dicionário de neurônios com sinal.
        if neuron2 not in neuron_signals:
            continue
                        
        # Verifica se há atividade no neurônio candidato.
        if neuron_signals[neuron2] <= 0:
            continue
            
        # Verifica se ainda não há conexão entre estes dois neurônios.
        if neuron2 not in my_connections and neuron not in neuron_connections[neuron2]:
            # Se chegou até aqui, a conexão é criada e encerra o loop (uma conexão por loop apenas).
            my_connections.append(neuron2)          
            my_strengths[neuron2] = 1         
            return

    # Percorre todos os neurônios da região para tentar criar novas conexões.
    #for i in range1:
    #    for j in range2:
    #        for k in range3:
    #            # Antes de começar qualquer coisa, faz um teste de probabilidade.
    #            if neuron_probs < np.random.rand():
    #                continue
    #            
    #            neuron2 = i + j*a + k*a*b
    #            
    #            # Verifica se o neurônio está no dicionário de neurônios com sinal.
    #            if neuron2 not in neuron_signals:
    #                continue
    #                            
    #            # Verifica se há atividade no neurônio candidato.
    #            if neuron_signals[neuron2] <= 0:
    #                continue
    #                
    #            # Verifica se ainda não há conexão entre estes dois neurônios.
    #            if neuron2 not in my_connections and neuron not in neuron_connections[neuron2]:
    #                # Se chegou até aqui, a conexão é criada e encerra o loop (uma conexão por loop apenas).
    #                my_connections.append(neuron2)          
    #                my_strengths[neuron2] = 1
    #                
    #                if npc_id == 'blob_0':
    #                    print(f'{npc_id}: Criou nova conexão!!! {neuron} ===> {neuron2}')   
    #             
    #                return
                        
    return
                        
def update_connections_weakening(npc_id, e_orig, e_now, neuron):
    npc = NPC[npc_id]
    neuron_outputs = npc['outputs']
    neuron_intputs = npc['inputs']
    neuron_connections = npc['connections']
    neuron_connection_strengths = npc['connection_strengths']
    my_connections = neuron_connections[neuron]
    my_strengths = neuron_connection_strengths[neuron]
    
    for neuron2 in list(e_now[neuron].keys()): 
        if neuron2 not in neuron_outputs or neuron2 not in neuron_intputs:
            continue
            
        e = e_now[neuron][neuron2] - e_orig**2
        
        if e < 0:
            my_connections.remove(neuron2) 
            del my_strengths[neuron2]
        else:
            my_strengths[neuron2] = e            
        
    return 
    
def update_connections_strengthening(npc_id, e, neuron, neuron2):
    NPC[npc_id]['connection_strengths'][neuron][neuron2] = e
        
    return 
    
def process_npc_neurons_updates(npc_id, neuron_signals, neuron_signals_tmp, to_zero):
    npc = NPC[npc_id]
    neuron_signals = npc['neuron_signals']
    neuron_connections = npc['connections']
    
    # Zera os sinais dos neurônios que acabaram de enviar sinal.
    for neuron in to_zero:    
        neuron_signals[neuron] = 0
              
    # Atualiza os sinais recebidos por cada neurônio. Tem que aplicar essa parte depois de zerar os sinais pois um
    # neurônio que enviou sinal e ficou zerado pode ter recebido sinal de outro neurônio logo depois.    
    for neuron in neuron_signals_tmp:
       neuron_signals[neuron] = neuron_signals_tmp[neuron]
        
    npc['num_connections'] = sum([len(neuron_connections[neuron]) for neuron in neuron_connections])
    
    if it_real % N == 0 or it_real < N: 
        num_connections_history.append(f"{npc_id} {it} {npc['num_connections']}\n")
            
    return 
    
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
    neuron_outputs = NPC[npc_id]['outputs']
    neuron_signals = npc['neuron_signals']
    neuron_outputs_thr = npc['outputs_thr']
    
    # Propaga os sinais dos neurônios finais para os outputs.
    outputs = [0 for i in range(len(neuron_outputs))]
    exp_factor = npc['exp_response']

    for i, neuron in enumerate(neuron_outputs):            
        if neuron in neuron_signals:
            signal, thr = neuron_signals[neuron], neuron_outputs_thr[neuron]
            outputs[i] = 1 if exp_response(signal, exp_factor) > thr else 0
        else:
            outputs[i] = 0
            
    return outputs
                
# Inicia o servidor WebSocket local
if __name__ == "__main__":
    # uvicorn.run(app, host="127.0.0.1", port=8000) # Inicia o servidor na porta 8000
    config = uvicorn.Config(app, host="127.0.0.1", port=8000)
    server = uvicorn.Server(config)
    server.run()