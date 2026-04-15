# gui.py (your PyQt6 GUI code, modified)
import sys
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSizePolicy,
    QTextEdit,
    QDialog,
    QSpacerItem,
)
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QBrush, QPixmap
from PyQt6.QtCore import Qt, QTimer, QThread, QPointF
from qamica import Assistant

BG_COLOR = QColor("#1E1E1E")
TEXT_COLOR = QColor("#D4D4D4")
USER_WAVE_COLOR = QColor("#6A9955")
AI_WAVE_COLOR = QColor("#569CD6")
ACCENT_COLOR = QColor("#C586C0")
LOGO_TEXT_COLOR = QColor("#E0E0E0")
PLACEHOLDER_CIRCLE_COLOR = QColor("#505050")


class RecordingIndicatorWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self.is_active = False
        self.current_color = USER_WAVE_COLOR
        self.pulse_factor = 2
        self.pulse_direction = 1  # 1 for expand, -1 for shrink

        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._update_pulse)
        self.animation_timer.timeout.connect(self.update)  # Trigger repaint

    def _update_pulse(self):
        if not self.is_active:
            return

        # Simple pulsing effect
        pulse_speed = 0.01
        if self.pulse_direction == 1:
            self.pulse_factor += pulse_speed
            if self.pulse_factor >= 1.2:  # Max expansion
                self.pulse_factor = 1.2
                self.pulse_direction = -1
        else:
            self.pulse_factor -= pulse_speed
            if self.pulse_factor <= 0.8:  # Max shrinkage
                self.pulse_factor = 0.8
                self.pulse_direction = 1
        self.update()

    def set_active(self, active, color=USER_WAVE_COLOR):
        self.is_active = active
        self.current_color = color
        if active:
            if not self.animation_timer.isActive():
                self.pulse_factor = 1.0  # Reset pulse
                self.pulse_direction = 1
                self.animation_timer.start(50)  # Pulse update interval (20fps)
        else:
            self.animation_timer.stop()
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        widget_rect = self.rect()
        center_x = widget_rect.width() / 2
        center_y = widget_rect.height() / 2
        base_radius = (
            min(widget_rect.width(), widget_rect.height()) * 0.3
        )  # Smaller base radius

        if not self.is_active:
            pen = QPen(PLACEHOLDER_CIRCLE_COLOR, 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawEllipse(
                QPointF(center_x, center_y), base_radius, base_radius
            )
            return

        # Draw pulsing circle
        current_radius = base_radius * self.pulse_factor
        painter.setPen(Qt.PenStyle.NoPen)  # No outline for the main pulse

        # Optional: Create a gradient or softer look
        # For simplicity, a solid fill with opacity changes
        fill_color = QColor(self.current_color)
        # Make it slightly transparent and vary opacity with pulse for a softer feel
        # Opacity based on how far it is from the base_radius
        opacity_factor = (
            1.0 - abs(self.pulse_factor - 1.0) * 1.5
        )  # Stronger opacity near base, fades at extremes
        opacity_factor = max(0.3, min(1.0, opacity_factor))  # Clamp opacity
        fill_color.setAlphaF(opacity_factor)

        painter.setBrush(QBrush(fill_color))
        painter.drawEllipse(
            QPointF(center_x, center_y), current_radius, current_radius
        )

        # Optional: a static outer ring for reference
        pen_outer = QPen(self.current_color.darker(120), 1)
        painter.setPen(pen_outer)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(
            QPointF(center_x, center_y), base_radius * 1.25, base_radius * 1.25
        )


# --- ChatHistoryDialog (reuse from your GUI code, no changes needed if it was okay) ---
class ChatHistoryDialog(QDialog):
    # ... (Your existing ChatHistoryDialog code) ...
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Chat History")
        self.setMinimumSize(400, 500)
        self.setStyleSheet(
            f"QDialog {{ background-color: {BG_COLOR.name()}; }}"
        )

        layout = QVBoxLayout(self)
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet(
            f"""
            QTextEdit {{
                background-color: #252526;
                color: {TEXT_COLOR.name()};
                border: 1px solid #333333;
                font-size: 14px;
            }}
        """
        )
        layout.addWidget(self.chat_display)

    def update_history(self, history_data):
        self.chat_display.clear()
        html_output = ""
        for speaker, text in history_data:
            color = (
                AI_WAVE_COLOR.name()
                if speaker == "ai"
                else USER_WAVE_COLOR.name()
            )
            # Basic HTML escaping for text to prevent issues if text contains < or >
            import html

            escaped_text = html.escape(text)
            html_output += f"<p><b style='color:{color};'>{speaker.capitalize()}:</b> {escaped_text}</p>"
        self.chat_display.setHtml(html_output)
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )


class VoiceAssistantGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Amica - GUI")
        self.setGeometry(100, 100, 700, 600)  # Slightly adjusted size
        self.setStyleSheet(f"background-color: {BG_COLOR.name()};")

        self.assistant_thread = QThread()
        self.assistant = Assistant()
        self.assistant.moveToThread(self.assistant_thread)

        self.assistant.status_update_signal.connect(self.handle_status_update)
        self.assistant.listening_state_changed_signal.connect(
            self.handle_listening_state_changed
        )
        self.assistant.processing_update_signal.connect(
            self.handle_processing_update
        )
        self.assistant.user_text_ready_signal.connect(self.handle_user_text)
        self.assistant.ai_text_ready_signal.connect(self.handle_ai_text)
        self.assistant.ai_is_speaking_signal.connect(
            self.handle_ai_is_speaking
        )
        self.assistant.chat_history_updated_signal.connect(
            self.handle_chat_history_update
        )
        self.assistant_thread.started.connect(
            self.assistant.run_main_loop
        )  # Crucial: run loop in thread
        self.assistant_thread.start()

        self.chat_history_dialog = None
        self._init_ui()
        self.is_manually_listening = False  # Track GUI button state

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)  # Add some spacing between elements

        # --- Top Bar (Logo and Chat Button) ---
        top_bar_layout = QHBoxLayout()
        # ... (your existing top_bar_layout code) ...
        self.logo_label = QLabel()
        pixmap = QPixmap("logo.png")
        pixmap = pixmap.scaled(
            50,
            50,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.logo_label.setPixmap(pixmap)
        self.logo_label.setFixedSize(50, 50)
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # self.logo_label.setFont(QFont("Arial", 28, QFont.Weight.Bold))
        # self.logo_label.setStyleSheet(f"color: {LOGO_TEXT_COLOR.name()};")
        # self.logo_label.setFixedSize(50, 50)
        # self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_bar_layout.addWidget(self.logo_label)
        top_bar_layout.addSpacerItem(
            QSpacerItem(
                40,
                20,
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Minimum,
            )
        )
        self.chat_button = QPushButton("📜 Chat")
        self.chat_button.setStyleSheet(
            f"""
            QPushButton {{ background-color: {ACCENT_COLOR.name()}; color: white; border: none; padding: 8px 15px; font-size: 14px; border-radius: 5px;}}
            QPushButton:hover {{ background-color: {ACCENT_COLOR.lighter(120).name()}; }}
        """
        )
        self.chat_button.clicked.connect(self.toggle_chat_history)
        top_bar_layout.addWidget(self.chat_button)
        main_layout.addLayout(top_bar_layout)

        # --- Recording Indicator ---
        self.recording_indicator = (
            RecordingIndicatorWidget()
        )  # Use the new widget
        main_layout.addWidget(self.recording_indicator, 1)  # Stretch factor

        # --- Transcribed User Text Label ---
        self.user_text_title_label = QLabel("You said:")
        self.user_text_title_label.setFont(
            QFont("Arial", 12, QFont.Weight.Bold)
        )
        self.user_text_title_label.setStyleSheet(
            f"color: {USER_WAVE_COLOR.name()}; margin-top: 10px;"
        )
        main_layout.addWidget(self.user_text_title_label)

        self.user_text_display_label = QLabel("...")
        self.user_text_display_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self.user_text_display_label.setWordWrap(True)
        self.user_text_display_label.setFont(QFont("Arial", 14))
        self.user_text_display_label.setStyleSheet(
            f"color: {TEXT_COLOR.name()}; padding: 5px; background-color: #252526; border-radius: 5px;"
        )
        self.user_text_display_label.setMinimumHeight(50)  # Ensure some space
        self.user_text_display_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding
        )
        main_layout.addWidget(self.user_text_display_label)

        # --- LLM Response Text Label (was self.ai_text_label) ---
        self.ai_text_title_label = QLabel("Assistant:")
        self.ai_text_title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.ai_text_title_label.setStyleSheet(
            f"color: {AI_WAVE_COLOR.name()}; margin-top: 10px;"
        )
        main_layout.addWidget(self.ai_text_title_label)

        self.ai_response_display_label = QLabel(
            "..."
        )  # Renamed from self.ai_text_label
        self.ai_response_display_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self.ai_response_display_label.setWordWrap(True)
        self.ai_response_display_label.setFont(QFont("Arial", 14))
        self.ai_response_display_label.setStyleSheet(
            f"color: {TEXT_COLOR.name()}; padding: 5px; background-color: #252526; border-radius: 5px;"
        )
        self.ai_response_display_label.setMinimumHeight(
            50
        )  # Ensure some space
        self.ai_response_display_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding
        )
        main_layout.addWidget(self.ai_response_display_label)

        # Status/Loading Label (Can be overlaid or a dedicated spot)
        self.status_label = QLabel(
            ""
        )  # For "Listening...", "Transcribing..." etc.
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        iarial = QFont("Arial", 12)
        iarial.setItalic(True)
        self.status_label.setFont(iarial)
        self.status_label.setStyleSheet(f"color: {ACCENT_COLOR.name()};")
        self.status_label.setFixedHeight(25)  # Fixed height for status bar
        main_layout.addWidget(self.status_label)

        self.setLayout(main_layout)
        self.clear_text_displays()  # Clear text on startup

    def clear_text_displays(self):
        self.user_text_display_label.setText("...")
        self.ai_response_display_label.setText("...")
        self.status_label.setText("Assistant Ready")

    # --- SLOTS for Assistant Signals (Modified) ---
    def handle_listening_state_changed(self, is_listening):
        # self.is_manually_listening = is_listening
        if is_listening:
            self.recording_indicator.set_active(True, USER_WAVE_COLOR)
            self.status_label.setText("Listening...")
            self.user_text_display_label.setText(
                "..."
            )  # Clear previous user text
            self.ai_response_display_label.setText(
                "..."
            )  # Clear previous AI response
        else:
            self.recording_indicator.set_active(False)
            if not self.status_label.text().startswith(
                "Thinking"
            ) and not self.status_label.text().startswith("Transcribing"):
                self.status_label.setText("Idle")

    def handle_processing_update(self, message):
        if message:
            self.status_label.setText(message)
            self.recording_indicator.set_active(
                False
            )  # Indicator off during backend processing
        else:  # Empty message means processing done
            # self.status_label.setText("Processing complete.") # Or clear it
            pass

    def handle_user_text(self, transcription_result: dict):
        if transcription_result["language"] == "en":
            user_text = transcription_result["english"]
        else:
            user_text = transcription_result["tamil"]
        if "error" in user_text.lower():
            self.user_text_display_label.setText(
                f"<i style='color:red;'>{user_text}</i>"
            )
        else:
            self.user_text_display_label.setText(user_text)
        print(f"GUI User Text: {user_text}")
        # self.status_label.setText("User text received.") # Optional

    def handle_ai_text(self, text):
        self.ai_response_display_label.setText(text)
        print(f"GUI AI Text: {text}")
        self.status_label.setText("Assistant response ready.")  # Update status

    def handle_ai_is_speaking(self, is_speaking):
        if is_speaking:
            self.recording_indicator.set_active(
                True, AI_WAVE_COLOR
            )  # AI color for indicator
            self.status_label.setText("Assistant speaking...")
        else:
            self.recording_indicator.set_active(False)
            self.status_label.setText("Assistant finished speaking.")

    def toggle_manual_listening_gui(self):
        if (
            not self.is_manually_listening
        ):  # If button was "Listen", now try to start
            self.assistant.request_manual_listen()
            # The actual button text/state change will happen via listening_state_changed signal
        else:  # If button was "Stop", now try to process
            self.assistant.request_force_process()
            # Button text/state change will also happen via listening_state_changed

    # --- SLOTS for Assistant Signals ---
    def handle_status_update(self, message):
        print(f"GUI Status: {message}")
        # Optionally display some statuses in ai_text_label if not a processing message
        # For example: if "daemon started" or "loop finished"

    def handle_chat_history_update(self, history):
        if self.chat_history_dialog and self.chat_history_dialog.isVisible():
            self.chat_history_dialog.update_history(history)

    def toggle_chat_history(self):
        # ... (Your existing toggle_chat_history method, ensure it uses self.assistant.chat_history) ...
        if not self.chat_history_dialog:
            self.chat_history_dialog = ChatHistoryDialog(self)

        if self.chat_history_dialog.isVisible():
            self.chat_history_dialog.hide()
        else:
            self.chat_history_dialog.update_history(
                self.assistant.chat_history
            )  # Get latest from assistant
            self.chat_history_dialog.show()
            self.chat_history_dialog.raise_()
            self.chat_history_dialog.activateWindow()

    def closeEvent(self, event):
        print("Closing GUI, stopping assistant...")
        self.assistant.stop_assistant_processing()  # Signal the assistant's loop to stop
        self.assistant_thread.quit()
        if not self.assistant_thread.wait(5000):  # Wait up to 5s
            print("Assistant thread did not terminate gracefully, forcing...")
            self.assistant_thread.terminate()  # Force if necessary
            self.assistant_thread.wait()

        if self.chat_history_dialog:
            self.chat_history_dialog.accept()  # Or close()
        super().closeEvent(event)


if __name__ == "__main__":
    import subprocess

    #
    # english server
    subprocess.Popen(
        "python portals/stt_fast.py --model_size small --task transcribe",
        shell=True,
    )
    # tamil server
    subprocess.Popen(
        "python portals/stt_jax.py --model_size small --variant tamil",
        shell=True,
    )

    try:
        app = QApplication(sys.argv)
        gui = VoiceAssistantGUI()
        gui.show()
        sys.exit(app.exec())
    except Exception as _:
        app.closeAllWindows()
        gui.close()
