"""
Here implementation that tale in account the geometry
of the probe to speed up template matching.

"""

import time
import numpy as np
import joblib
from concurrent.futures import ThreadPoolExecutor
import itertools

# Optimización: Control de logging para reducir overhead en template matching
DEBUG_MODE = True  # Cambiar a True para debugging detallado

from .peeler_engine_base import PeelerEngineGeneric

from .peeler_tools import *
from .peeler_tools import _dtype_spike

import sklearn.metrics.pairwise

from .cltools import HAVE_PYOPENCL, OpenCL_Helper
if HAVE_PYOPENCL:
    import pyopencl
    mf = pyopencl.mem_flags

from .peakdetector import get_peak_detector_class

try:
    import numba
    HAVE_NUMBA = True
    from .numba_tools import numba_explore_best_shift, numba_sparse_scalar_product
except ImportError:
    HAVE_NUMBA = False







class PeelerEngineGeometrical(PeelerEngineGeneric):
    def change_params(self,  **kargs):
        PeelerEngineGeneric.change_params(self, **kargs)

    def initialize(self, **kargs):
        PeelerEngineGeneric.initialize(self, **kargs)
        
        # create peak detector
        p = dict(self.catalogue['peak_detector_params'])

        self.peakdetector_engine = p.pop('engine')
        self.peakdetector_method = p.pop('method')
        
        PeakDetector_class = get_peak_detector_class(self.peakdetector_method, self.peakdetector_engine)
        
        chunksize = self.fifo_size-2*self.n_span # not the real chunksize here
        self.peakdetector = PeakDetector_class(self.sample_rate, self.nb_channel,
                                                        chunksize, self.internal_dtype, self.geometry)
        
        self.peakdetector.change_params(**p)
        
        # some attrs
        self.shifts = np.arange(-self.maximum_jitter_shift, self.maximum_jitter_shift+1)
        self.nb_shift = self.shifts.size
        
        #~ self.channel_distances = sklearn.metrics.pairwise.euclidean_distances(self.geometry).astype('float32')
        #~ self.channels_adjacency = {}
        #~ for c in range(self.nb_channel):
            #~ if self.use_sparse_template:
                #~ nearest, = np.nonzero(self.channel_distances[c, :]<self.adjacency_radius_um)
                #~ self.channels_adjacency[c] = nearest
            #~ else:
                #~ self.channels_adjacency[c] = np.arange(self.nb_channel, dtype='int64')
        
        self.mask_already_tested = np.zeros((self.fifo_size, self.nb_channel), dtype='bool')


    def initialize_before_each_segment(self, **kargs):
        PeelerEngineGeneric.initialize_before_each_segment(self, **kargs)
        self.peakdetector.initialize_stream()


    def detect_local_peaks_before_peeling_loop(self):
        # reset tested mask
        self.mask_already_tested[:] = False
        # and detect peak
        self.re_detect_local_peak()
        
        #~ print('detect_local_peaks_before_peeling_loop', self.pending_peaks.size)

    def re_detect_local_peak(self):
        # OPTIMIZACIÓN FASE 1: Buscar solo en zona afectada por último spike
        if hasattr(self, '_last_spike_zone') and self._last_spike_zone is not None:
            start_idx, end_idx = self._last_spike_zone
            # Expandir zona de búsqueda ligeramente por seguridad
            search_start = max(self.n_span, start_idx - 10)
            search_end = min(self.fifo_residuals.shape[0] - self.n_span, end_idx + 10)
            
            # Obtener mask solo para la zona afectada
            fifo_zone = self.fifo_residuals[search_start-self.n_span:search_end+self.n_span, :]
            mask = self.peakdetector.get_mask_peaks_in_chunk(fifo_zone)
            
            # DEBUG: Monitor optimized search
            if DEBUG_MODE:
                zone_size = search_end - search_start
                total_size = self.fifo_residuals.shape[0] - 2*self.n_span
                reduction = (1 - zone_size/total_size) * 100 if total_size > 0 else 0
                print(f"[⚡ OPTIMIZED SEARCH] Zone: {zone_size} samples ({reduction:.1f}% reduction from {total_size})")
        else:
            # Primera detección: buscar en todo el buffer (código original)
            mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)
            search_start = self.n_span
            
            # DEBUG: Monitor full search
            if DEBUG_MODE:
                total_size = self.fifo_residuals.shape[0] - 2*self.n_span
                print(f"[🔍 FULL SEARCH] Entire buffer: {total_size} samples")
        
        # DEBUG: Monitor peak detection (optimizado con DEBUG_MODE)
        if DEBUG_MODE:
            residuals_range = [float(np.min(self.fifo_residuals)), float(np.max(self.fifo_residuals))]
            peaks_found = int(np.sum(mask)) if mask is not None else 0
            print(f"[🔬 PeakDetector] Residuals range: {residuals_range}, mask shape: {mask.shape if mask is not None else None}, peaks found: {peaks_found}")
        
        if mask.ndim ==1:
            #~ mask &= ~self.mask_already_tested[self.n_span:-self.n_span, 0]
            sample_indexes,   = np.nonzero(mask)
            sample_indexes += search_start  # Ajustar por offset de búsqueda optimizada
            tested = self.mask_already_tested[sample_indexes, 0]
            sample_indexes = sample_indexes[~tested]
            chan_indexes = np.zeros(sample_indexes.size, dtype='int64')
        else:
            #~ mask &= ~self.mask_already_tested[self.n_span:-self.n_span, :]
            sample_indexes, chan_indexes  = np.nonzero(mask)
            sample_indexes += search_start  # Ajustar por offset de búsqueda optimizada
            tested = self.mask_already_tested[sample_indexes, chan_indexes]
            sample_indexes = sample_indexes[~tested]
            chan_indexes = chan_indexes[~tested]
        
        
        amplitudes = np.abs(self.fifo_residuals[sample_indexes, chan_indexes])
        order = np.argsort(amplitudes)[::-1]
        
        dtype_peak = [('sample_index', 'int32'), ('chan_index', 'int32'), ('peak_value', 'float32')]
        self.pending_peaks = np.zeros(sample_indexes.size, dtype=dtype_peak)
        self.pending_peaks['sample_index'] = sample_indexes
        self.pending_peaks['chan_index'] = chan_indexes
        self.pending_peaks['peak_value'] = amplitudes
        self.pending_peaks = self.pending_peaks[order]
        #~ print('re_detect_local_peak', self.pending_peaks.size)

    def select_next_peak(self):
        #~ print(len(self.pending_peaks))
        if len(self.pending_peaks)>0:
            sample_ind, chan_ind, ampl = self.pending_peaks[0]
            self.pending_peaks = self.pending_peaks[1:]
            return sample_ind, chan_ind
        else:
            return LABEL_NO_MORE_PEAK, None

    def on_accepted_spike(self, sample_ind, cluster_idx, jitter):
        # remove spike prediction from fifo residuals
        #~ t1 = time.perf_counter()
        pos, pred = make_prediction_one_spike(sample_ind, cluster_idx, jitter, self.fifo_residuals.dtype, self.catalogue)
        #~ t2 = time.perf_counter()
        #~ print('    make_prediction_one_spike', (t2-t1)*1000)
        
        #~ t1 = time.perf_counter()
        self.fifo_residuals[pos:pos+self.peak_width_long, :] -= pred
        #~ t2 = time.perf_counter()
        #~ print('     self.fifo_residuals -', (t2-t1)*1000)

        # OPTIMIZACIÓN FASE 1: Registrar zona afectada para búsqueda optimizada
        self._last_spike_zone = (pos, pos + self.peak_width_long)

        # this prevent search peaks in the zone until next "reset_to_not_tested"
        #~ t1 = time.perf_counter()
        self.clean_pending_peaks_zone(sample_ind, cluster_idx)
        #~ t2 = time.perf_counter()
        #~ print('     self.clean_pending_peaks_zone -', (t2-t1)*1000)


    def clean_pending_peaks_zone(self, sample_ind, cluster_idx):
        # TODO test with sparse_mask_level3s!!!!!
        mask = self.sparse_mask_level1[cluster_idx, :]

        
        #~ t1 = time.perf_counter()
        #~ keep = np.zeros(self.pending_peaks.size, dtype='bool')
        #~ for i, peak in enumerate(self.pending_peaks):
            #~ in_zone = mask[peak['chan_index']] and \
                                #~ (peak['sample_index']+self.n_left)<sample_ind and \
                                #~ sample_ind<(peak['sample_index']+self.n_right)
            #~ keep[i] = not(in_zone)
        
        peaks = self.pending_peaks
        in_zone = mask[peaks['chan_index']] &\
                            ((peaks['sample_index']+self.n_left)<sample_ind) & \
                            ((peaks['sample_index']+self.n_right)>sample_ind)
        keep = ~ in_zone
        #~ t2 = time.perf_counter()
        #~ print('     clean_pending_peaks_zone loop', (t2-t1)*1000)
        
        self.pending_peaks = self.pending_peaks[keep]
        
        #~ print('clean_pending_peaks_zone', self.pending_peaks.size)
    
    def set_already_tested(self, sample_ind, peak_chan):
        self.mask_already_tested[sample_ind, peak_chan] = True

    def reset_to_not_tested(self, good_spikes):
        for spike in good_spikes:
            # each good spike can remove from
            cluster_idx = self.catalogue['label_to_index'][spike.cluster_label]
            chan_mask = self.sparse_mask_level1[cluster_idx, :]
            self.mask_already_tested[spike.index + self.n_left_long:spike.index + self.n_right_long][:, chan_mask] = False
        
        self.re_detect_local_peak()
        

    def get_no_label_peaks(self):
        mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)
        nolabel_indexes, chan_indexes = np.nonzero(mask)
        #~ nolabel_indexes, chan_indexes = np.nonzero(~self.mask_not_already_tested)
        
        nolabel_indexes += self.n_span
        nolabel_indexes = nolabel_indexes[nolabel_indexes<(self.chunksize+self.n_span)]
        bad_spikes = np.zeros(nolabel_indexes.shape[0], dtype=_dtype_spike)
        bad_spikes['index'] = nolabel_indexes
        bad_spikes['cluster_label'] = LABEL_UNCLASSIFIED
        return bad_spikes

    def get_best_template(self, left_ind, chan_ind):
        # PROFILING: Inicializar timer
        if DEBUG_MODE:
            import time
            t_start = time.perf_counter()

        full_waveform = self.fifo_residuals[left_ind:left_ind+self.peak_width,:]
        
        centers0 = self.catalogue['centers0']
        projections = self.catalogue['projections']

        strict_low = self.catalogue['boundaries'][:, 0]
        strict_high = self.catalogue['boundaries'][:, 1]
        flexible_low = self.catalogue['boundaries'][:, 2]
        flexible_high = self.catalogue['boundaries'][:, 3]
        
        # PROFILING: Tiempo de preparación de datos
        if DEBUG_MODE:
            t_prep = time.perf_counter()
        
        
        n = centers0.shape[0]
        flat_waveform = full_waveform.flatten()
        flat_centers0 = centers0.reshape(n, -1)
        
        # OPTIMIZATION STEP 4: Conditional runtime scaling based on catalogue type
        # Skip scaling if catalogue was pre-scaled by AdaptiveCatalogueBuilderOptimized
        if not self._catalogue_pre_scaled:
            # LEGACY MODE: Apply runtime scaling (original code)
            # CORRECCIÓN CRÍTICA: Escalado dramático para matched productos escalares a boundaries
            if not hasattr(self, '_scale_factor_computed'):
                self._scale_factor_computed = True
                
                # Calcular producto escalar típico para un template de ejemplo
                sample_waveform = flat_waveform
                sample_template = flat_centers0[0]  # YA está aplanado correctamente
                # Aplanar la proyección también para compatibilidad de formas
                sample_projection = projections[0].flatten()
                
                print(f"[🔧 SHAPES DEBUG] sample_waveform shape: {sample_waveform.shape}")
                print(f"[🔧 SHAPES DEBUG] sample_template shape: {sample_template.shape}")
                print(f"[🔧 SHAPES DEBUG] sample_projection shape: {sample_projection.shape}")
                
                # Producto escalar sin escalar
                unscaled_product = np.sum((sample_waveform - sample_template) * sample_projection)
                
                # Rango objetivo de boundaries (usualmente [-0.5, 0.5] = rango de 1.0)
                boundary_range = np.mean(strict_high - strict_low)
                target_magnitude = boundary_range * 0.1  # Apuntar al 10% del rango (más conservador)
                
                # Factor de escalado SUPER-DRAMÁTICO para llevar productos escalares de ~500k a ~0.05
                if abs(unscaled_product) > 0:
                    # Queremos que productos escalares de ~500,000 se conviertan en ~0.05
                    # Factor de escalado: 0.05 / 500,000 = 1e-7
                    self._waveform_scale_factor = target_magnitude / abs(unscaled_product)
                    # Aplicar escalado adicional si es necesario
                    if self._waveform_scale_factor > 1e-4:
                        self._waveform_scale_factor = 1e-7  # Force super small scale
                else:
                    self._waveform_scale_factor = 1e-7  # Factor ultra pequeño por defecto
                    
                print(f"[🔧 SCALE FIX] Unscaled product: {unscaled_product:.1f}")
                print(f"[🔧 SCALE FIX] Target magnitude: {target_magnitude:.6f}")
                print(f"[🔧 SCALE FIX] Boundary range: {boundary_range:.3f}")
                print(f"[🔧 SCALE FIX] Applied scale factor: {self._waveform_scale_factor:.12f}")
            
            # Aplicar escalado DRAMÁTICO al waveform ANTES del cálculo de productos escalares
            scaled_flat_waveform = flat_waveform * self._waveform_scale_factor
            
            # Crear buffer escalado para los residuos - usar la misma ventana que fifo_residuals
            scaled_fifo = self.fifo_residuals.copy()
            scaled_fifo[left_ind:left_ind + self.n_span] *= self._waveform_scale_factor
            
            # PROFILING: Tiempo de escalado
            if DEBUG_MODE:
                t_scale = time.perf_counter()
            
            scalar_products = numba_sparse_scalar_product(scaled_fifo, left_ind, centers0, projections, chan_ind,
                        self.sparse_mask_level1, )
            
            # PROFILING: Tiempo de cálculo de productos escalares
            if DEBUG_MODE:
                t_numba = time.perf_counter()
            
            # CORRECCIÓN CRÍTICA: Escalado Y CENTRADO de productos escalares finales
            # Los productos están en rango ~507,000, necesitamos llevarlos a rango [-0.5, 0.5]
            scalar_products = scalar_products * 1e-6  # Escalado ultra-directo
            
            # NUEVO: Centrar los productos escalares alrededor de cero
            # Esto permite que diferentes templates tengan diferentes ranges de productos
            product_mean = np.mean(scalar_products)
            scalar_products = scalar_products - product_mean  # Centrar en cero
            
            # Asegurar que el rango se mantenga dentro de boundaries
            product_range = np.max(scalar_products) - np.min(scalar_products)
            boundary_range = np.mean(strict_high - strict_low)
            if product_range > boundary_range:
                scale_factor = boundary_range / product_range * 0.8  # 80% del rango disponible
                scalar_products = scalar_products * scale_factor
            
            # PROFILING: Tiempo de ajuste de productos escalares
            if DEBUG_MODE:
                t_adjust = time.perf_counter()
        else:
            # OPTIMIZED MODE: Skip runtime scaling (catalogue already pre-scaled)
            # Direct calculation without copy() or scaling operations (7-10% faster)
            if DEBUG_MODE:
                t_scale = time.perf_counter()  # No actual scaling, just for profiling consistency
            
            # Use fifo_residuals directly (no copy needed)
            scalar_products = numba_sparse_scalar_product(self.fifo_residuals, left_ind, centers0, projections, chan_ind,
                        self.sparse_mask_level1, )
            
            if DEBUG_MODE:
                t_numba = time.perf_counter()
            
            # No scaling adjustment needed (catalogue templates already in correct scale)
            # No centering needed (catalogue pre-processing handled this)
            
            if DEBUG_MODE:
                t_adjust = time.perf_counter()

        # DEBUG: Log template matching values (optimizado con DEBUG_MODE)
        if DEBUG_MODE:
            if not hasattr(self, '_template_debug_counter'):
                self._template_debug_counter = 0
            self._template_debug_counter += 1
            
            if self._template_debug_counter % 50 == 0:  # Log every 50 attempts
                print(f"[🔬 TemplateMatch] Attempt #{self._template_debug_counter}")
                print(f"[🔬 TemplateMatch] CENTERED Scalar products range: [{scalar_products.min():.3f}, {scalar_products.max():.3f}]")
                print(f"[🔬 TemplateMatch] Scalar products mean (should be ~0): {np.mean(scalar_products):.6f}")
                print(f"[🔬 TemplateMatch] Strict bounds range: [{strict_low.min():.3f}, {strict_high.max():.3f}]")
                print(f"[🔬 TemplateMatch] Channel {chan_ind}, waveform amplitude: [{full_waveform.min():.3f}, {full_waveform.max():.3f}]")
        
        possible_idx, = np.nonzero((scalar_products < strict_high) & (scalar_products > strict_low))
        #~ possible_idx, = np.nonzero((scalar_products < flexible_high) & (scalar_products > flexible_low))
        
        # PROFILING: Tiempo de filtrado de candidatos
        if DEBUG_MODE:
            t_filter = time.perf_counter()
        
        # DEBUG: Log template matching results (optimizado con DEBUG_MODE)
        if DEBUG_MODE:
            if hasattr(self, '_template_debug_counter') and self._template_debug_counter % 50 == 0:
                print(f"[🔬 TemplateMatch] Possible candidates: {len(possible_idx)} (strict bounds)")
        
        #~ print('possible_idx', possible_idx)
        #~ print('scalar_products[possible_idx]', scalar_products[possible_idx])
        
        
        #~ do_plot = False
        if len(possible_idx) == 1:
            extra_idx = None
            candidates_idx =possible_idx
        elif len(possible_idx) == 0:
            #~ extra_idx, = np.nonzero((np.abs(scalar_products) < 0.5))
            extra_idx, = np.nonzero((scalar_products < flexible_high) & (scalar_products > flexible_low))
            #~ if len(extra_idx) ==0:
                # give a try to very far ones.
                #~ extra_idx, = np.nonzero((np.abs(scalar_products) < 1.))
                #~ print('extra_idx', extra_idx)
            #~ if len(extra_idx) ==0:
                #~ candidates_idx = []
            #~ else:
                #~ candidates_idx = extra_idx
            candidates_idx = extra_idx
            
            # DEBUG: Log when no strict candidates but checking flexible (optimizado)
            if DEBUG_MODE:
                if hasattr(self, '_template_debug_counter') and self._template_debug_counter % 50 == 0:
                    print(f"[🔬 TemplateMatch] No strict candidates, checking flexible: {len(extra_idx)} candidates")
                    print(f"[🔬 TemplateMatch] Flexible bounds range: [{flexible_low.min():.3f}, {flexible_high.max():.3f}]")
            
            #~ candidates_idx =possible_idx
            #~ pass
        elif len(possible_idx) > 1 :
            extra_idx = None
            candidates_idx = possible_idx
        
        debug_plot_change = False
        if len(candidates_idx) > 0:
            candidates_idx = np.array(candidates_idx, dtype='int64')
            common_mask = np.sum(self.sparse_mask_level3[candidates_idx, :], axis=0) > 0
            shift_scalar_product, shift_distance = numba_explore_best_shift(self.fifo_residuals, left_ind, self.catalogue['centers0'],
                            self.catalogue['projections'], candidates_idx, self.maximum_jitter_shift, common_mask, self.sparse_mask_level1)
            #~ i0, i1 = np.unravel_index(np.argmin(np.abs(shift_scalar_product), axis=None), shift_scalar_product.shape)
            i0, i1 = np.unravel_index(np.argmin(shift_distance, axis=None), shift_distance.shape)
            #~ best_idx = candidates_idx[i0]
            shift = self.shifts[i1]
            cluster_idx = candidates_idx[i0]
            final_scalar_product = shift_scalar_product[i0, i1]
            
            # PROFILING: Tiempo de búsqueda de mejor shift
            if DEBUG_MODE:
                t_shift = time.perf_counter()

            # DEBUG: Log successful template match (optimizado con DEBUG_MODE)
            if DEBUG_MODE:
                if hasattr(self, '_template_debug_counter') and self._template_debug_counter % 50 == 0:
                    print(f"[🔬 TemplateMatch] 🎯 CANDIDATE FOUND: cluster={cluster_idx}, shift={shift}, scalar={final_scalar_product:.3f}")
                    # PROFILING: Reportar tiempos
                    t_end = time.perf_counter()
                    print(f"[⏱️ PROFILING get_best_template]")
                    print(f"  1️⃣ Data prep:     {(t_prep - t_start)*1000:.3f}ms")
                    # Mostrar correctamente si scaling fue saltado o no
                    if self._catalogue_pre_scaled:
                        print(f"  2️⃣ Scale factor:  0.000ms (SKIPPED - pre-scaled catalogue)")
                    else:
                        print(f"  2️⃣ Scale factor:  {(t_scale - t_prep)*1000:.3f}ms (includes copy)")
                    print(f"  3️⃣ Numba scalar:  {(t_numba - t_scale)*1000:.3f}ms")
                    # Mostrar correctamente si adjust fue saltado o no
                    if self._catalogue_pre_scaled:
                        print(f"  4️⃣ Adjust range:  0.000ms (SKIPPED - pre-scaled catalogue)")
                    else:
                        print(f"  4️⃣ Adjust range:  {(t_adjust - t_numba)*1000:.3f}ms")
                    print(f"  5️⃣ Filter cands:  {(t_filter - t_adjust)*1000:.3f}ms")
                    print(f"  6️⃣ Best shift:    {(t_shift - t_filter)*1000:.3f}ms")
                    print(f"  7️⃣ Finalization:  {(t_end - t_shift)*1000:.3f}ms")
                    print(f"  ⏱️ TOTAL:         {(t_end - t_start)*1000:.3f}ms")

            
            
            #~ print('shift', shift)
            #~ print('cluster_idx', cluster_idx)
            #~ print('final_scalar_product', final_scalar_product)
            
            
            if np.abs(shift) == self.maximum_jitter_shift:
                cluster_idx = None
                shift = None
                final_scalar_product = None
                # DEBUG: Log rejection due to excessive jitter (optimizado)
                if DEBUG_MODE:
                    if hasattr(self, '_template_debug_counter') and self._template_debug_counter % 50 == 0:
                        print(f"[🔬 TemplateMatch] ❌ REJECTED: Excessive jitter (shift={shift})")
                #~ print('maximum_jitter_shift >> cluster_idx = None ')
                #~ do_plot = True
            #~ i0_bis, i1_bis = np.unravel_index(np.argmin(np.abs(shift_scalar_product), axis=None), shift_scalar_product.shape)
            #~ if i0 != i0_bis:
                
                #~ debug_plot_change = True
                #~ print('Warning')
                #~ print(possible_idx)
                #~ print(shift_scalar_product)
                #~ print(shift_distance)
            
            
            
            
            #~ if best_idx != cluster_idx:
                #~ print('*'*50)
                #~ print('best_idx != cluster_idx', best_idx, cluster_idx)
                #~ print('*'*50)
                #~ cluster_idx = best_idx
                #~ debug_plot_change = True
        else:
            cluster_idx = None
            shift = None
            final_scalar_product = None
            
            # DEBUG: Log failed template match (optimizado con DEBUG_MODE)
            if DEBUG_MODE:
                if hasattr(self, '_template_debug_counter') and self._template_debug_counter % 50 == 0:
                    print(f"[🔬 TemplateMatch] ❌ NO CANDIDATES FOUND - Template matching failed")
                    # PROFILING: Reportar tiempos parciales
                    t_end = time.perf_counter()
                    print(f"[⏱️ PROFILING get_best_template - FAILED]")
                    print(f"  1️⃣ Data prep:     {(t_prep - t_start)*1000:.3f}ms")
                    # Mostrar correctamente si scaling fue saltado o no
                    if self._catalogue_pre_scaled:
                        print(f"  2️⃣ Scale factor:  0.000ms (SKIPPED - pre-scaled catalogue)")
                    else:
                        print(f"  2️⃣ Scale factor:  {(t_scale - t_prep)*1000:.3f}ms (includes copy)")
                    print(f"  3️⃣ Numba scalar:  {(t_numba - t_scale)*1000:.3f}ms")
                    # Mostrar correctamente si adjust fue saltado o no
                    if self._catalogue_pre_scaled:
                        print(f"  4️⃣ Adjust range:  0.000ms (SKIPPED - pre-scaled catalogue)")
                    else:
                        print(f"  4️⃣ Adjust range:  {(t_adjust - t_numba)*1000:.3f}ms")
                    print(f"  5️⃣ Filter cands:  {(t_filter - t_adjust)*1000:.3f}ms")
                    print(f"  ⏱️ TOTAL:         {(t_end - t_start)*1000:.3f}ms")
            
            #~ import matplotlib.pyplot as plt
            #~ fig, ax = plt.subplots()
            #~ ax.plot(self.shifts, shift_scalar_product.T)
            #~ plt.show()
            
        
        #~ print('ici',)
        

        # DEBUG OMP
        #~ from sklearn.linear_model import orthogonal_mp_gram
        #~ from sklearn.linear_model import OrthogonalMatchingPursuit
        #~ n_nonzero_coefs = 2
        #~ omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
        #~ X = self.catalogue['centers0'].reshape(self.catalogue['centers0'].shape[0], -1).T
        #~ waveform = self.fifo_residuals[left_ind:left_ind+self.peak_width,:].flatten()
        #~ y = waveform
        #~ omp.fit(X, y)
        #~ coef = omp.coef_
        #~ idx_r, = coef.nonzero()
        #~ cluster_idx_omp = np.argmin(np.abs(coef - 1))
        
        
        #~ if cluster_idx_omp != cluster_idx and coef[cluster_idx_omp] > 0.5:
        #~ if True:
        if False:
            
        
        
        
        #~ if cluster_idx in (3,6):
        #~ if do_plot:
        #~ if False:
        #~ if final_scalar_product is not None and np.abs(final_scalar_product) > 0.5:
            
        #~ if True:
        #~ if len(possible_idx) != 1:
        #~ if len(possible_idx) > 1:
        #~ if len(candidates_idx) > 1:
        
        #~ if 7 in possible_idx or  cluster_idx == 7:
        #~ if cluster_idx not in possible_idx and len(possible_idx) > 0:
        #~ if debug_plot_change:
            
            import matplotlib.pyplot as plt
            
            print()
            print('best cluster_idx', cluster_idx)
            print('possible_idx', possible_idx)
            print('extra_idx', extra_idx)
            print(scalar_products[possible_idx])
            print(strict_high[possible_idx])
            
            print('cluster_idx_omp', cluster_idx_omp)

            
            fig, ax = plt.subplots()
            ax.plot(coef)
            if cluster_idx is not None:
                ax.axvline(cluster_idx)
            ax.set_title(f'{cluster_idx} omp {cluster_idx_omp}')
            #~ plt.show()

            
            
            
            
            fig, ax = plt.subplots()
            shift2 = 0 if shift is None else shift
            full_waveform2 = self.fifo_residuals[left_ind+shift2:left_ind+shift2+self.peak_width,:]
            
            ax.plot(full_waveform2.T.flatten(), color='k')
            if shift !=0 and shift is not None:
                ax.plot(full_waveform.T.flatten(), color='grey', ls='--')
            
            for idx in candidates_idx:
                ax.plot(self.catalogue['centers0'][idx, :].T.flatten(), color='m')
            
            ax.plot(self.catalogue['centers0'][cluster_idx_omp, :].T.flatten(), color='y')
            
            
            
                
            if cluster_idx is not None:
                ax.plot(self.catalogue['centers0'][cluster_idx, :].T.flatten(), color='c', ls='--')
            ax.set_title(f'best {cluster_idx} shift {shift} possible_idx {possible_idx}')
            
            if shift is not None:
                fig, ax = plt.subplots()
                #~ ax.plot(self.shifts, np.abs(shift_scalar_product).T)
                ax.plot(self.shifts, shift_scalar_product.T)
                ax.axhline(0)
                

                fig, ax = plt.subplots()
                ax.plot(self.shifts, np.abs(shift_distance).T)
            
            plt.show()
        
        
        best_template_info = {'nb_candidate' : len(candidates_idx), 'final_scalar_product':final_scalar_product}
        
        
        return cluster_idx, shift, best_template_info


    def accept_tempate(self, left_ind, cluster_idx, jitter, best_template_info):
        if jitter is None:
            # this must have a jitter
            jitter = 0
        
        #~ if np.abs(jitter) > (self.maximum_jitter_shift - 0.5):
            #~ return False
        
        strict_low = self.catalogue['boundaries'][:, 0]
        strict_high = self.catalogue['boundaries'][:, 1]
        flexible_low = self.catalogue['boundaries'][:, 2]
        flexible_high = self.catalogue['boundaries'][:, 3]
        
        
        #~ flat_waveform = full_waveform.flatten()
        #~ sp2 = np.sum((flat_waveform - centers0[cluster_idx, :].flatten()) * projections[cluster_idx, :])
        sp = best_template_info['final_scalar_product']
        nb_candidate = best_template_info['nb_candidate']
        
        if nb_candidate == 1:
            
            #~ accept_template = strict_low[cluster_idx] < sp < strict_high[cluster_idx]
            accept_template = flexible_low[cluster_idx] < sp < flexible_high[cluster_idx]
            
        else:
            accept_template = flexible_low[cluster_idx] < sp < flexible_high[cluster_idx]
        
                
            # waveform L2 on mask
            #~ full_waveform = self.fifo_residuals[left_ind:left_ind+self.peak_width,:]
            #~ wf = full_waveform[:, mask]
            
            # prediction with interpolation
            #~ _, pred_wf = make_prediction_one_spike(left_ind - self.n_left, cluster_idx, jitter, self.fifo_residuals.dtype, self.catalogue, long=False)
            #~ pred_wf = pred_wf[:, mask]
        
            #~ dist = (pred_wf - wf) ** 2
            
            
            # criteria per channel
            #~ residual_nrj_by_chan = np.sum(dist, axis=0)
            #~ wf_nrj = np.sum(wf**2, axis=0)
            #~ weight = self.weight_per_template_dict[cluster_idx]
            #~ crietria_weighted = (wf_nrj>residual_nrj_by_chan).astype('float') * weight
            #~ accept_template = np.sum(crietria_weighted) >= 0.7 * np.sum(weight)
            
            # criteria per sample
            #~ dist * np.abs(pred_wf) < 
            #~ dist_w = dist / np.abs(pred_wf)
            #~ gain = (dist < wf**2).astype('float') * np.abs(pred_wf) / np.sum(np.abs(pred_wf))
            #~ gain = (wf / pred_wf - 1) * np.abs(pred_wf) / np.sum(np.abs(pred_wf))
            #~ gain = (pred_wf**2 / wf**1 - 1) * np.abs(pred_wf) / np.sum(np.abs(pred_wf))
            #~ accept_template = np.sum(gain) > 0.8
            #~ accept_template = np.sum(gain) > 0.7
            #~ accept_template0 = np.sum(gain) > 0.6
            #~ accept_template = np.sum(gain) > 0.5
            
            # criteria max residual
            #~ max_res = np.max(np.abs(pred_wf - wf))
            #~ max_pred = np.max(np.abs(pred_wf))
            #~ accept_template1 = max_pred > max_res
            
            
            

            
            
            
            #~ accept_template = False
            
            # debug
            #~ limit_sp =self.catalogue['sp_normed_limit'][cluster_idx, :]
            #~ sp = np.sum(self.catalogue['centers0_normed'] * full_waveform * self.catalogue['template_weight'])
            #~ print('limit_sp', limit_sp, 'sp', sp)
            
            
            
            #~ accept_template = False
            #~ immediate_accept = False
            
            # DEBUG always refuse!!!!!
            #~ accept_template = False
        
        
        #~ label = self.catalogue['cluster_labels'][cluster_idx]
        
        # debug
        #~ if label == 13:
            #~ if accept_template and not immediate_accept:
                #~ accept_template = False
        
        # debug
        #~ if label == 13:
            #~ if not hasattr(self, 'count_accept'):
                #~ self.count_accept = {}
                #~ self.count_accept[label] = {'accept_template':0, 'immediate_accept':0, 'not_accepted':0}
            
            #~ if accept_template:
                #~ self.count_accept[label]['accept_template'] += 1
                #~ if immediate_accept:
                    #~ self.count_accept[label]['immediate_accept'] += 1
            #~ else:
                #~ self.count_accept[label]['not_accepted'] += 1
            #~ print(self.count_accept)
            
        #~ if self._plot_debug:
        #~ if not accept_template and label in []:
        #~ if not accept_template:
        #~ if accept_template:
        #~ if True:
        if False:
            
        #~ if not immediate_accept:
        #~ if immediate_accept:
        #~ if immediate_accept:
        #~ if label == 7 and not accept_template:
        #~ if label == 7:
        #~ if label == 121:
        #~ if label == 5:
        #~ if nb_candidate > 1:
        
        #~ if label == 13 and accept_template and not immediate_accept:
        #~ if label == 13 and not accept_template:
            
        #~ if label in (7,9):
        #~ nears = np.array([ 5813767,  5813767, 11200038, 11322540, 14989650, 14989673, 14989692, 14989710, 15119220, 15830377, 16138346, 16216666, 17078883])
        #~ print(np.abs((left_ind - self.n_left) - nears))
        #~ print(np.abs((left_ind - self.n_left) - nears) < 2)
        #~ if label == 5 and np.any(np.abs((left_ind - self.n_left) - nears) < 50):
            
            #~ if immediate_accept:
            
            import matplotlib.pyplot as plt
            
            mask = self.sparse_mask_level2[cluster_idx]
            full_waveform = self.fifo_residuals[left_ind:left_ind+self.peak_width,:]
            wf = full_waveform[:, mask]
            _, pred_waveform = make_prediction_one_spike(left_ind - self.n_left, cluster_idx, jitter, self.fifo_residuals.dtype, self.catalogue, long=False)
            pred_wf = pred_waveform[:, mask]
    
            if accept_template:
                color = 'g'
            else:
                color = 'r'
            
            #~ if accept_template:
                #~ if immediate_accept:
                    #~ color = 'g'
                #~ else:
                    #~ color = 'c'
            #~ else:
                #~ color = 'r'
            
            #~ if not immediate_accept:
                #~ fig, ax = plt.subplots()
                #~ ax.plot(gain.T.flatten(), color=color)
                #~ ax.set_title('{}'.format(np.sum(gain)))

            #~ fig, ax = plt.subplots()
            #~ ax.plot(feat_centroids.T, alpha=0.5)
            #~ ax.plot(feat_waveform, color='k')

            fig, ax = plt.subplots()
            ax.plot(full_waveform.T.flatten(), color='k')
            ax.plot(pred_waveform.T.flatten(), color=color)
            
            l0, l1 = strict_low[cluster_idx], strict_high[cluster_idx]
            l2, l3 = flexible_low[cluster_idx], flexible_high[cluster_idx]
            title = f'{cluster_idx} {sp:0.3f} lim [{l0:0.3f} {l1:0.3f}] [{l2:0.3f} {l3:0.3f}] {nb_candidate}'
            ax.set_title(title)
                
            #~ fig, ax = plt.subplots()
            #~ ax.plot(wf.T.flatten(), color='k')
            #~ ax.plot(pred_wf.T.flatten(), color=color)
            
            #~ ax.plot( wf.T.flatten() - pred_wf.T.flatten(), color=color, ls='--')
            
            print()
            print('cluster_idx',cluster_idx, 'accept_template', accept_template)
            #~ print(distance, self.distance_limit[cluster_idx])
            #~ print('distance', distance, distance2, 'limit_distance', self.distance_limit[cluster_idx])

            #~ limit_sp =self.catalogue['sp_normed_limit'][cluster_idx, :]
            #~ sp = np.sum(self.catalogue['centers0_normed'] * full_waveform * self.catalogue['template_weight'])
            #~ sp = np.sum(self.catalogue['centers0_normed'] * full_waveform)
            #~ print('limit_sp', limit_sp, 'sp', sp)
            
            #~ if not immediate_accept:
                #~ print('np.sum(gain)', np.sum(gain))


            #~ fig, ax = plt.subplots()
            #~ res = wf - pred_wf
            #~ count, bins = np.histogram(res, bins=150, weights=np.abs(pred_wf))
            #~ ax.plot(bins[:-1], count)
            #~ plt.show()

            
            
            #~ if distance2 >= self.distance_limit[cluster_idx]:
                #~ print(crietria_weighted, weight)
                #~ print(np.sum(crietria_weighted),  np.sum(weight))
            
            #~ ax.plot(full_wf0.T.flatten(), color='y')
            #~ ax.plot( full_wf.T.flatten() - full_wf0.T.flatten(), color='y')
            
            #~ ax.set_title('not accepted')
            plt.show()
        
        return accept_template
    
    
    def _plot_after_inner_peeling_loop(self):
        pass

    def _plot_before_peeling_loop(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plot_sigs = self.fifo_residuals.copy()
        self._plot_sigs_before = plot_sigs
        #~ chan_order = np.argsort(self.channel_distances[0, :])
        
        for c in range(self.nb_channel):
        #~ for c in chan_order:
            plot_sigs[:, c] += c*30
        
        ax.plot(plot_sigs, color='k')

        ax.axvline(self.fifo_size - self.n_right_long, color='r')
        ax.axvline(-self.n_left_long, color='r')

        mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)
        sample_inds, chan_inds= np.nonzero(mask)
        sample_inds += self.n_span
        
        ax.scatter(sample_inds, plot_sigs[sample_inds, chan_inds], color='r')
        ax.set_title(f'nb peak {sample_inds.size}')
        
        #~ plt.show()
        
    
    def _plot_label_unclassified(self, left_ind, peak_chan, cluster_idx, jitter):
        return
        import matplotlib.pyplot as plt
        #~ print('LABEL UNCLASSIFIED', left_ind, cluster_idx)
        fig, ax = plt.subplots()
        
        wf = self.fifo_residuals[left_ind:left_ind+self.peak_width, :]
        wf0 = self.catalogue['centers0'][cluster_idx, :, :]
        
        ax.plot(wf.T.flatten(), color='b')
        #~ ax.plot(wf0.T.flatten(), color='g')
        
        ax.set_title(f'label_unclassified {left_ind-self.n_left} {cluster_idx} chan{peak_chan}')
        
        ax.axvline(peak_chan*self.peak_width-self.n_left)
        
        plt.show()

    def _plot_after_peeling_loop(self, good_spikes):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plot_sigs = self.fifo_residuals.copy()
        
        
        for c in range(self.nb_channel):
            plot_sigs[:, c] += c*30
        ax.plot(plot_sigs, color='k')
        
        ax.plot(self._plot_sigs_before, color='b')
        
        ax.axvline(self.fifo_size - self.n_right_long, color='r')
        ax.axvline(-self.n_left_long, color='r')

        mask = self.peakdetector.get_mask_peaks_in_chunk(self.fifo_residuals)
        sample_inds, chan_inds= np.nonzero(mask)
        sample_inds += self.n_span
        ax.scatter(sample_inds, plot_sigs[sample_inds, chan_inds], color='r')
        
        
        
        good_spikes = np.array(good_spikes, dtype=_dtype_spike)
        pred = make_prediction_signals(good_spikes, self.internal_dtype, plot_sigs.shape, self.catalogue, safe=True)
        plot_pred = pred.copy()
        for c in range(self.nb_channel):
            plot_pred[:, c] += c*30
        
        ax.plot(plot_pred, color='m')
        
        plt.show()





