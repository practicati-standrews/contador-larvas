import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np
import io
import config
from pathlib import Path

# --- Configuración de la Página ---
st.set_page_config(
    page_title=config.APP_NAME,
    page_icon=config.LOGO_PATH,
    layout="wide",
)

# --- CSS personalizado ---
st.markdown(
    """
    <style>
    .metric-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 15px;
        padding: 20px;
        font-size: 1.5rem;
    }
    .metric-label {
        font-weight: bold;
        font-size: 1.5rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# --- Funciones auxiliares ---
@st.cache_resource
def load_model(model_path):
    """
    Carga el modelo YOLO y lo guarda en caché para no recargarlo en cada ejecución.
    """
    try:
        model = YOLO(str(model_path))
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None


def process_image(image, model, confidence):
    """
    Procesa la imagen pura
    """
    # Detección YOLO
    results = model(image, conf=confidence, max_det=10000)
    count = len(results[0].boxes)
    # Dibujar resultados
    res_bgr = results[0].plot(line_width=1, font_size=1, labels=False)
    # Convertir a RGB para Streamlit
    res_rgb = np.ascontiguousarray(res_bgr[..., ::-1])  # Convertir BGR a RGB
    # Añadir texto de conteo
    text = f"TOTAL: {count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Obtener dimensiones para escalar texto
    h, w, _ = res_rgb.shape
    font_scale = max(1, w / 1000)
    thickness = max(2, int(font_scale * 2))
    # Colores
    color_bg = (27, 77, 137)
    color_text = (255, 255, 255)
    # Obtener tamaño del texto
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    # Coordenadas del recuadro
    margin = int(10 * font_scale)
    box_coords = ((0, 0), (text_w + margin * 2, text_h + margin * 2))
    cv2.rectangle(res_rgb, box_coords[0], box_coords[1], color_bg, -1)
    text_x = margin
    text_y = text_h + margin - int(baseline / 2)
    cv2.putText(
        res_rgb, text, (text_x, text_y), font, font_scale, color_text, thickness
    )
    return res_rgb, count


# --- Interfaz Principal ---
def main():
    # Sidebar (barra lateral)
    with st.sidebar:
        if Path(config.LOGO_PATH).exists():
            st.image(str(config.LOGO_PATH), width=200)

        st.header("Configuración")

        # Selector de confianza
        confidence = st.slider(
            "Nivel de Confianza",
            min_value=config.MIN_CONFIDENCE,
            max_value=config.MAX_CONFIDENCE,
            value=config.DEFAULT_CONFIDENCE,
            step=0.05,
            help="Ajusta el nivel de confianza para la detección de larvas.",
        )

        st.info(f"Version: {config.VERSION}\n\n{config.COMPANY_NAME}")

    # Header principal
    st.title(config.APP_NAME)
    st.markdown(
        f"**{config.COMPANY_NAME}** - Sistema de detección de larvas de chorito asistido por IA."
    )
    st.divider()

    # Cargar modelo
    model = load_model(config.DEFAULT_MODEL_PATH)

    if model is None:
        st.warning(
            f"No se encontró el modelo en: {config.DEFAULT_MODEL_PATH}. Verifique la ruta."
        )
        st.stop()

    # Cargar imagen
    col_upload1, col_upload2, col_upload3 = st.columns([1, 2, 1])
    with col_upload2:
        uploaded_file = st.file_uploader(
            "Cargar Imagen",
            type=["jpg", "jpeg", "png"],
            help="Sube una imagen para detectar y contar larvas de chorito.",
        )

    # Verificar si la imagen cambió o se eliminó
    if uploaded_file is not None:
        current_file_name = uploaded_file.name
        # Si hay un archivo previo y es diferente, limpiar resultados
        if "previous_file_name" in st.session_state:
            if st.session_state["previous_file_name"] != current_file_name:
                # Limpiar resultados previos
                keys_to_clear = ["result_image", "count"]
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]

        # Guardar el nombre del archivo actual
        st.session_state["previous_file_name"] = current_file_name
    else:
        # Si no hay archivo, limpiar todo
        keys_to_clear = ["previous_file_name", "result_image", "count"]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

    # Logica de deteccion
    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file)

        # Boton de procesamiento
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            detect_clicked = st.button(
                "Detectar Larvas", type="primary", use_container_width=True
            )
        st.divider()

        if detect_clicked:
            with st.spinner("Procesando imagen..."):
                result_image, count = process_image(image_pil, model, confidence)
                # Guardar en session_state
                st.session_state["result_image"] = result_image
                st.session_state["count"] = count

        # Métricas/Resultados y botón de descarga
        if "result_image" in st.session_state and "count" in st.session_state:
            metric_col1, metric_col2, metric_col3 = st.columns([1, 2, 1])
            with metric_col2:
                st.markdown(
                    f"""
                    <div class="metric-container">
                        <span class="metric-label">Larvas Detectadas:</span>
                        <span class="metric-value">{st.session_state["count"]}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            download_col1, download_col2, download_col3 = st.columns([1, 2, 1])
            with download_col2:
                # Preparar descarga de imagen resultante
                img_result_pil = Image.fromarray(st.session_state["result_image"])
                buf = io.BytesIO()
                img_result_pil.save(buf, format="JPEG", quiality=95)
                byte_im = buf.getvalue()

                original_name = Path(uploaded_file.name).stem
                new_filename = (
                    f"resultado_{original_name}_({st.session_state['count']}).jpg"
                )
                st.download_button(
                    label="Descargar Resultado",
                    data=byte_im,
                    file_name=new_filename,
                    mime="image/jpeg",
                    use_container_width=True,
                )
        st.divider()

        # Columnas: Original - Resultados
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Imagen Original")
            st.image(image_pil, use_container_width=True)

        with col2:
            st.subheader("Resultados Detección")

            # Mostrar imagen resultante si existe
            if "result_image" in st.session_state:
                st.image(st.session_state["result_image"], use_container_width=True)
            else:
                st.info("Presiona el botón 'Detectar Larvas' para ver los resultados.")

    else:
        st.info("Por favor, carga una imagen para comenzar la detección.")


if __name__ == "__main__":
    main()
