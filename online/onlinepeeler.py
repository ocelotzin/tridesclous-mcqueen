from ..peeler import Peeler


from pyqtgraph.util.mutex import Mutex

import pyacq
from pyacq import Node, register_node_type, ThreadPollInput

from ..peeler import _dtype_spike

# Optimización: Control de logging para reducir overhead en hot-paths
DEBUG_MODE = False  # Cambiar a True para debugging detallado




class PeelerThread(ThreadPollInput):
    def __init__(self, input_stream, output_streams, peeler, in_group_channels, geometry, sample_rate, total_channel, timeout=200, parent=None):
        ThreadPollInput.__init__(self, input_stream,  timeout=timeout, return_data=True, parent=parent)
        self.output_streams = output_streams
        self.peeler = peeler
        self.in_group_channels = in_group_channels
        self.geometry = geometry
        self.sample_rate = sample_rate
        self.total_channel = total_channel
        self.mutex = Mutex()
    def process_data(self, pos, sigs_chunk):
        # LOGGING AGREGADO PARA DEBUG (optimizado con DEBUG_MODE)
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
            self._spike_total = 0
            if DEBUG_MODE:
                print(f"[🔍 PeelerThread] Iniciando logging de datos...")
        
        self._debug_counter += 1
        
        try:
            assert sigs_chunk.shape[0] == self.peeler.chunksize, f'PeelerThread chunksize is BAD!! {sigs_chunk.shape[0]} {self.peeler.chunksize}'
        except AssertionError as e:
            return
        with self.mutex:
            sigs_chunk = sigs_chunk[:, self.in_group_channels]
            sig_index, preprocessed_chunk, total_spike, spikes  = self.peeler.process_one_chunk(pos, sigs_chunk)
            
            # LOG CADA 100 CHUNKS (solo si DEBUG_MODE=True)
            if DEBUG_MODE and self._debug_counter % 100 == 0:
                print(f"[🔍 PeelerThread] Chunk #{self._debug_counter}: shape={sigs_chunk.shape}, spikes={len(spikes)}, amp_range=[{sigs_chunk.min():.2f}, {sigs_chunk.max():.2f}]")
            
            if spikes.size > 0:
                self._spike_total += len(spikes)
                if DEBUG_MODE:
                    print(f"[🎯 PeelerThread] 🎉 ¡SPIKES DETECTADOS! Count: {len(spikes)}, Total: {self._spike_total}")
                pass
            else:
                pass
            
            # Call callbacks if they exist (for wrapper nodes)
            if hasattr(self, '_signal_callback') and callable(self._signal_callback):
                if DEBUG_MODE:
                    print(f"[🔍 PeelerThread] Calling _signal_callback with chunk shape={preprocessed_chunk.shape}")
                self._signal_callback(preprocessed_chunk, sig_index)
            else:
                if DEBUG_MODE:
                    print(f"[❌ PeelerThread] No _signal_callback available")
                
            if spikes.size > 0:
                if hasattr(self, '_spike_callback') and callable(self._spike_callback):
                    if DEBUG_MODE:
                        print(f"[🎯 PeelerThread] Calling _spike_callback with {len(spikes)} spikes, total_spike={total_spike}")
                    self._spike_callback(spikes, total_spike)
                    if DEBUG_MODE:
                        print(f"[✅ PeelerThread] _spike_callback completed successfully")
                else:
                    # MANTENER: Este es un error crítico, siempre debe mostrarse
                    print(f"[❌ PeelerThread] CRITICAL: {len(spikes)} spikes detected but NO _spike_callback available!")
                    print(f"    - hasattr _spike_callback: {hasattr(self, '_spike_callback')}")
                    print(f"    - callable _spike_callback: {callable(getattr(self, '_spike_callback', None))}")
            else:
                if DEBUG_MODE:
                    if hasattr(self, '_spike_callback'):
                        print(f"[🔍 PeelerThread] No spikes in chunk (callback available)")
                    else:
                        print(f"[🔍 PeelerThread] No spikes in chunk (no callback configured)")
        
    
    def change_params(self, **peeler_params):
        #~ print('PeelerThread.change_params')
        with self.mutex:
            # complicated hack pad_width can be changed on the fly
            # because it break the index stream with advance or delay
            # so we always forec the prvious one!!!!
            if hasattr(self.peeler, 'peeler_engine') and hasattr(self.peeler.peeler_engine, 'signalpreprocessor'):
                #~ print(self.peeler.peeler_engine.signalpreprocessor.pad_width)
                prev_pad_width = self.peeler.peeler_engine.signalpreprocessor.pad_width
                #~ if 'pad_width' in catalogue['signal_preprocessor_params']:
                peeler_params['catalogue']['signal_preprocessor_params']['pad_width'] = prev_pad_width
                #~ print('*'*50)
                #~ print('Force prev_pad_width', prev_pad_width)
                #~ print('*'*50)
            else:
                prev_pad_width = None

            self.peeler.change_params(**peeler_params)
            
            buffer_spike_index = self.output_streams['spikes'].last_index
            #~ print('buffer_spike_index', buffer_spike_index)
            
            # TODO check tha pad_width have not changed bechause
            # head index will be out
            
            
            
            self.peeler.initialize_online_loop(sample_rate=self.sample_rate,
                                                nb_channel=len(self.in_group_channels),
                                                source_dtype=self.input_stream().params['dtype'],
                                                geometry=self.geometry,
                                                )
            
            #~ print('*'*50)
            #~ print(self.peeler.peeler_engine.signalpreprocessor.pad_width)
            #~ print('*'*50)
            
            #~ print('self.peeler.peeler_engine.total_spike', self.peeler.peeler_engine.total_spike)
            self.peeler.peeler_engine.total_spike = buffer_spike_index
            #~ print('self.peeler.peeler_engine.total_spike', self.peeler.peeler_engine.total_spike)
            
            
            

class OnlinePeeler(Node):
    """
    Wrapper on top of Peeler class to make a pyacq Node.
    And so to have on line spike sorting!!
    """
    _input_specs = {'signals' : dict(streamtype = 'signals')}
    _output_specs = {'signals' : dict(streamtype = 'signals'),
                                'spikes': dict(streamtype='events', shape = (-1, ),  dtype=_dtype_spike),
                                }

    def __init__(self , **kargs):
        Node.__init__(self, **kargs)
    
    def _configure(self, in_group_channels=None, catalogue=None, chunksize=None,
                                    internal_dtype='float32', 
                                    geometry=None,
                                    **peeler_params):
        
        #~ if 'engine' in peeler_engine_kargs:
            #~ peeler_engine = peeler_engine_kargs.pop('engine')
        #~ print('peeler_engine', peeler_engine)
        
        assert geometry is not None
        
        self.in_group_channels = in_group_channels
        self.catalogue = catalogue
        self.chunksize = chunksize
        self.internal_dtype = internal_dtype
        
        self.geometry = geometry
        self.peeler_params = peeler_params
        
        

    def after_input_connect(self, inputname):
        self.total_channel = self.input.params['shape'][1]
        self.sample_rate = self.input.params['sample_rate']
        
        # internal dtype (for waveforms) will also be the output dtype
        self.outputs['signals'].spec['dtype'] = self.internal_dtype
        self.outputs['signals'].spec['shape'] = (-1, len(self.in_group_channels))
        self.outputs['signals'].spec['sample_rate'] = self.input.params['sample_rate']
    
    def after_output_configure(self, inputname):
        channel_info = self.input.params.get('channel_info', None)
        if channel_info is not None:
            channel_info = [channel_info[c] for c in self.in_group_channels]
            self.outputs['signals'].params['channel_info'] = channel_info
    
    def _initialize(self):
        self.peeler = Peeler(dataio=None)
        # Espera a que el stream tenga los parámetros necesarios
        for _ in range(50):  # Espera hasta 5 segundos (50*0.1s)
            print('[OnlinePeeler] Esperando params en input:', getattr(self.input, 'params', None))
            if hasattr(self.input, 'params') and 'sample_rate' in self.input.params and 'shape' in self.input.params:
                sample_rate = self.input.params['sample_rate']
                total_channel = self.input.params['shape'][1]
                break
            import time
            time.sleep(0.1)
        else:
            print('[OnlinePeeler] ERROR: params finales en input:', getattr(self.input, 'params', None))
            raise RuntimeError("El InputStream no tiene los parámetros 'sample_rate' y 'shape' disponibles en .params tras esperar. Asegúrate de que el stream esté correctamente conectado y configurado antes de inicializar el nodo.")
        self.thread = PeelerThread(self.input, self.outputs, self.peeler, self.in_group_channels, self.geometry, sample_rate, total_channel)
        self.change_catalogue(self.catalogue)
    def _start(self):
        self.thread.start()
    def _stop(self):
        self.thread.stop()
        self.thread.wait()

        
    def _close(self):
        pass
    
    def change_catalogue(self, catalogue):
        self.catalogue = catalogue
        self.thread.change_params(catalogue=catalogue, 
                                        chunksize=self.chunksize, internal_dtype=self.internal_dtype,
                                        **self.peeler_params)
                                        
                                        
        
    
    
register_node_type(OnlinePeeler)
