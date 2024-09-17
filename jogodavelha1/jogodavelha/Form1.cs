namespace jogodavelha
{
    public partial class Form1 : Form
    {
        int Xplayer = 0, Oplayer = 0, empatesPontos = 0, rodadas = 0;
        bool turno = true, jogo_final = false;
        string[] texto = new string[9];

        public Form1()
        {
            InitializeComponent();
        }

        private void btn_Click(object sender, EventArgs e)
        {
            Button btn = (Button)sender;
            int ButtonIndex = btn.TabIndex;

            if (btn.Text == "" && jogo_final == false)
            {
                if (turno)
                {
                    btn.Text = "X";
                    texto[ButtonIndex] = btn.Text;
                    rodadas++;
                    turno = !turno;
                    Checagem(1);
                }
                else
                {
                    btn.Text = "O";
                    texto[ButtonIndex] = btn.Text;
                    rodadas++;
                    turno = !turno;
                    Checagem(2);
                }
            }
        }

        void Vencedor(int PlayerQueGanhou)
        {
            jogo_final = true;
            if (PlayerQueGanhou == 1)
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

        void Checagem(int ChecagemPlayer)
        {
            string suporte = (ChecagemPlayer == 1) ? "X" : "O";

            // Verifica linhas horizontais
            for (int horizontal = 0; horizontal < 8; horizontal += 3)
            {
                if (suporte == texto[horizontal] && texto[horizontal] == texto[horizontal + 1] && texto[horizontal] == texto[horizontal + 2])
                {
                    Vencedor(ChecagemPlayer);
                    return;
                }
            }

            // Verifica colunas verticais
            for (int vertical = 0; vertical < 3; vertical++)
            {
                if (suporte == texto[vertical] && texto[vertical] == texto[vertical + 3] && texto[vertical] == texto[vertical + 6])
                {
                    Vencedor(ChecagemPlayer);
                    return;
                }
            }

            // Verifica diagonal principal
            if (texto[0] == suporte && texto[0] == texto[4] && texto[0] == texto[8])
            {
                Vencedor(ChecagemPlayer);
                return;
            }

            // Verifica diagonal secundária
            if (texto[2] == suporte && texto[2] == texto[4] && texto[2] == texto[6])
            {
                Vencedor(ChecagemPlayer);
                return;
            }

            // Verifica empate
            if (rodadas == 9 && jogo_final == false)
            {
                MessageBox.Show("Empate!");
                jogo_final = true;
            }
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

            // Limpa o array que guarda as jogadas
            for (int i = 0; i < 9; i++)
            {
                texto[i] = "";
            }
        }
    }
}
