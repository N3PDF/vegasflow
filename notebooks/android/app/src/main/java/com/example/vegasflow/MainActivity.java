package com.example.vegasflow;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.common.FileUtil;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;


public class MainActivity extends AppCompatActivity {

    protected Interpreter tflite;
    protected MappedByteBuffer tfliteModel;
    protected Interpreter.Options tfliteOptions;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        try {
            tfliteModel = FileUtil.loadMappedFile(this, "model.tflite");
            tfliteOptions = new Interpreter.Options();
        } catch (IOException ie)
        {
            ie.printStackTrace();
        }

        tflite = new Interpreter(tfliteModel, tfliteOptions);

        final EditText input = (EditText) findViewById(R.id.input);
        final Button clickButton = (Button) findViewById(R.id.button);
        final TextView result = (TextView) findViewById(R.id.result);

        clickButton.setOnClickListener( new View.OnClickListener() {

            @Override
            public void onClick(View v) {

                float[][] a = {{Float.parseFloat(input.getText().toString())}};
                float[] b = {0f};

                tflite.run(a, b);

                result.setText(String.valueOf(b[0]));
            }
        });

    }

    @Override
    protected void finalize()
    {
        tflite.close();
    }
}
