using System;
using System.Linq;
using System.Windows.Forms;

namespace jogodavelha
{
    public partial class Form1 : Form
    {
        int rodadas = 0;
        bool turno = true, jogo_final = false; // true = X, false = O

        public Form1()
        {
            InitializeComponent();
        }

        private void btn_Click(object sender, EventArgs e)
        {
            if (jogo_final) return;

            Button btn = (Button)sender;
            if (btn.Text == "" && turno)
            {
                // Jogada do jogador X
                btn.Text = "X";
                rodadas++;
                Checagem("X");

                // Se o jogo não terminou, a IA joga
                if (!jogo_final)
                {
                    turno = !turno;
                    Application.DoEvents(); // Permite a atualização da interface
                    JogarIA();
                }
            }
        }

        private void JogarIA()
        {
            // Escolhe uma jogada aleatória
            Random rand = new Random();
            int index;
            do
            {
                index = rand.Next(0, 9);
            } while (GetButtonByIndex(index).Text != "");

            // Acessa o botão diretamente usando o índice
            Button btn = GetButtonByIndex(index);
            if (btn != null)
            {
                // Faz a jogada
                btn.Text = "O";
                rodadas++;
                Checagem("O");
            }
            else
            {
                MessageBox.Show("Botão não encontrado.");
            }

            // Alterna o turno
            turno = !turno;
        }

        private Button GetButtonByIndex(int index)
        {
            // Retorna o botão com base no índice
            switch (index)
            {
                case 0: return button1;
                case 1: return button2;
                case 2: return button3;
                case 3: return button4;
                case 4: return button5;
                case 5: return button6;
                case 6: return button7;
                case 7: return button8;
                case 8: return button9;
                default: return null;
            }
        }

        void Vencedor(string jogador)
        {
            jogo_final = true;
            if (jogador == "X")
            {
                MessageBox.Show("Jogador X ganhou!");
                turno = true;
            }
            else
            {
                MessageBox.Show("Jogador O ganhou!");
                turno = false;
            }
        }

        void Checagem(string jogador)
        {
            // Verifica linhas horizontais
            if (CheckLine(button1, button2, button3, jogador) ||
                CheckLine(button4, button5, button6, jogador) ||
                CheckLine(button7, button8, button9, jogador))
            {
                Vencedor(jogador);
                return;
            }

            // Verifica colunas verticais
            if (CheckLine(button1, button4, button7, jogador) ||
                CheckLine(button2, button5, button8, jogador) ||
                CheckLine(button3, button6, button9, jogador))
            {
                Vencedor(jogador);
                return;
            }

            // Verifica diagonal principal
            if (CheckLine(button1, button5, button9, jogador) ||
                CheckLine(button3, button5, button7, jogador))
            {
                Vencedor(jogador);
                return;
            }

            // Verifica empate
            if (rodadas == 9 && !jogo_final)
            {
                MessageBox.Show("Empate!");
                jogo_final = true;
            }
        }

        bool CheckLine(Button btn1, Button btn2, Button btn3, string jogador)
        {
            return btn1.Text == jogador && btn1.Text == btn2.Text && btn1.Text == btn3.Text;
        }

        private void button10_Click(object sender, EventArgs e)
        {
            // Limpa todos os botões do tabuleiro
            button1.Text = "";
            button2.Text = "";
            button3.Text = "";
            button4.Text = "";
            button5.Text = "";
            button6.Text = "";
            button7.Text = "";
            button8.Text = "";
            button9.Text = "";

            // Reinicia o estado do jogo
            rodadas = 0;
            jogo_final = false;
            turno = true;
        }
    }
}
