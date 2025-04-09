#pragma once

#include <QApplication>
#include <QPushButton>
#include <QMainWindow>
#include <QVBoxLayout>
#include <QCommonStyle>
#include <QMouseEvent>
#include <QWidget>
#include <QLabel>
#include <QPixmap>
#include <QImage>
#include <QTimer>
#include <QDebug>
#include <QSignalMapper>
#include <QScreen>
#include <QMenu>
#include <QAction>

#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>

struct ViewerGeometry
{
    int x;
    int y;
    int w;
    int h;
};

enum InferenceType_e
{
    NONE_INFERENCE,
    SINGLE_INFERENCE_YOLOV8,
    // ...
};

/**
 * @class FrameViewer
 * @brief Responsible for displaying images coming from capture devices.
 * @details Each channel inside a screen is associated with a FrameViewer.
 */
class FrameViewer : public QWidget
{
    Q_OBJECT

public:
    bool running_;
    int idx_;
    InferenceType_e inf_type_;
    int display_frame_idx_;
    std::vector<cv::Mat *> display_frame_list_;
    float confidence_;

    FrameViewer(QWidget *parent, bool show_confidence);
    ~FrameViewer();
    uint32_t width();
    uint32_t height();
    void SetGeometry(int x, int y, int w, int h);
    void SetIdx(int idx);
    void UpdateFrame(cv::Mat *frame);
    void UpdateFPS(float fps);
    void HideFPS();
    void HideChannelName();
    void HideConfidence();
    void UpdateConfidence();
    void HideModelName();
    void UpdateModelName(const char* model_name);
signals:
    void SignalUpdateFrame(cv::Mat *);
    void SignalUpdateFPS(float);
    void SignalUpdateConfidence();

public slots:
    void SlotUpdateFrame(cv::Mat *frame);
    void SlotUpdateFPS(float);
    void SlotUpdateConfidence();
    void SlotConfAdd();
    void SlotConfReduce();
private:
    int x_{0}, y_{0}, w_{0}, h_{0};
    QLabel *frame_{nullptr};
    QLabel *name_{nullptr};
    QLabel *fps_{nullptr};
    QLabel *model_label_{nullptr};
    QLabel *resolution_label_{nullptr};
    QLabel *confidence_label_{nullptr};
    std::vector<QPushButton *> model_buttons_;
    QWidget *confidence_button_widget{nullptr};
    QPushButton *button_up{nullptr};
    QPushButton *button_down{nullptr};
};

/**
 * @class DisplayScreen
 * @brief Manages multiple FrameViewers for displaying streaming channels.
 */
class DisplayScreen : public QWidget
{
    Q_OBJECT

private:
    bool running_ = false;
    uint32_t w_, h_;
    uint32_t num_viewers_;
    std::vector<FrameViewer *> viewers_;
    std::vector<ViewerGeometry> viewer_geometry_;

public:
    DisplayScreen();

    /**
     * @brief Constructs a DisplayScreen with a parent widget and specific screen settings.
     * @param parent The parent QWidget.
     * @param qscreen The QScreen object for display settings.
     */
    DisplayScreen(QWidget *parent, QScreen *qscreen);

    /**
     * @brief Destructor for DisplayScreen, releasing resources.
     */
    ~DisplayScreen();

    /**
     * @brief Updates the display for a specific viewer with frame and FPS data.
     * @param viewer_id The index of the viewer to update.
     * @param frame The new frame to display.
     * @param fps The frames per second value to display.
     */
    void SetDisplayFrame(int viewer_id, cv::Mat *frame, float fps);
    
    /**
     * @brief Updates the display for a specific viewer with a new frame, without FPS data.
     * @param viewer_id The index of the viewer to update.
     * @param frame The new frame to display.
     */
    void SetDisplayFrame(int viewer_id, cv::Mat *frame);

    /**
     * @brief Updates the display for a specific viewer using a frame object.
     * @param viewer_id The index of the viewer to update.
     * @param frame The new frame to display.
     */
    void SetDisplayFrame(int viewer_id, cv::Mat frame);

    /**
     * @brief Retrieves the frame context buffer to be displayed for a viewer.
     * @param viewer_id The index of the viewer.
     * @return A pointer to the cv::Mat object representing the frame context buffer.
     */
    cv::Mat *GetDisplayFrameBuf(int viewer_id);

    /**
     * @brief Retrieves the width of the display.
     * @return The width in pixels.
     */
    uint32_t width();

    /**
     * @brief Retrieves the height of the display.
     * @return The height in pixels.
     */
    uint32_t height();

    /**
     * @brief Gets the width of a specific viewer's display.
     * @param viewer_id The index of the viewer.
     * @return The width in pixels.
     */
    uint32_t GetViewerWidth(int viewer_id);

    /**
     * @brief Gets the height of a specific viewer's display.
     * @param viewer_id The index of the viewer.
     * @return The height in pixels.
     */
    uint32_t GetViewerHeight(int viewer_id);

    /**
     * @brief Arranges viewers in a square layout based on the specified channel count.
     * @param num_channels The number of channels to display.
     */
    void SetSquareLayout(int num_channels);

    /**
     * @brief Retrieves the total number of viewers on the screen.
     * @return The number of viewers.
     */
    uint32_t NumViewers();

    /**
     * @brief Adds a new viewer to the display.
     * @param viewer Pointer to the FrameViewer object to add.
     */
    void AddViewer(FrameViewer *viewer);

    /**
     * @brief Updates the confidence value displayed.
     * @param confidence The confidence value to set.
     */
    void SetConfidenceValue(float confidence);

    /**
     * @brief Retrieves the current confidence value.
     * @return The confidence value.
     */
    float GetConfidenceValue();

    /**
     * @brief Updates the model name displayed on the screen.
     * @param model_name The new model name to display.
     */
    void SetModelName(std::string model_name);

    /**
     * @brief Displays the screen interface.
     */
    void Show();

signals:

public slots:
};

/**
 * @class YOLOGuiView
 * @brief Provides functionality to initialize and manage the GUI application lifecycle.
 */
class YOLOGuiView
{
public:
    /**
     * @brief Creates a YOLOGuiView instance and sets up screen configurations.
     * @param argc The number of command-line arguments.
     * @param argv The array of command-line argument strings.
     */
    YOLOGuiView(int &argc, char *argv[]);

    /**
     * @brief Cleans up resources used by YOLOGuiView.
     */
    ~YOLOGuiView();

    /**
     * @brief Launches the GUI and keeps the application running until it exits.
     * @return The application's exit status.
     */
    int Run();

    void Exit();

    /**
     * @brief Control handle of each screen object
     */
    std::vector<std::unique_ptr<DisplayScreen>> screens;

private:
    QApplication app;
};
