const N_LAYERS: usize = 96; // Amount of transformer layers 
const D_MODEL: usize = 12288; // Embedding size/hidden dimension
const D_MLP: usize = 4 * D_MODEL; // Dimension of MLP 
const D_HEAD: usize = 128; // Dimension of each attention head
const N_HEADS: usize = D_MODEL / D_HEAD; // Number of heads
const N_VOCAB: usize = 50000; //  Number of distinct tokens in vocab

type Token = u64; 
type Logits = [f32; N_VOCAB];

trait ARModel {
    fn apply(&self, tokens: &[Token]) -> Vec<Logits>;
}

// This defines a trait in Rust called ARModel. This defines behavior that types can implement
// Any type implementing "ARModel" must define the function apply 

/*
&self: takes an immutable reference to the object (self)

tokens: &[Token]: takes a slice of Token (i.e. a borrowed view into a list of tokens)

-> Vec<Logits>: returns a Vec of Logits (likely the output scores from a model)
*/

// #[derive(Clone)]

// Defining the state/residual vector 
struct State([f32; D_MODEL]);

// At every point inside the model, we have one State per token
// position

//This is a slice because you want dynamic length 
type ResidualStream = [State]; 

// Query vector
type Query = State; 

// Update Vector; Added to the state vector 
type Update = State; 

impl State{
    fn zero() -> Self {
        State([0.0; D_MODEL])
    }

    fn update(&self, right: &Update) -> State {
        let mut out = self.clone(); 
        for (i, r) in right.0.iter().enumerate(){
            out.0[i] += r;
        }
        out
    }

    fn query(&self, right: &Query) -> f32 {
        dot(&self.0, &right.0)
    }
}

// <> specifies your parameters 
// I think & means reference/pointer? 
// out probably means the thing you return 
// Querying something means dotting a vector with the query vector to create a single floating-point value. 
fn dot<const N: usize>(l: &[f32; N], r: &[f32; N]) -> f32{
    let mut out = 0.0;
    for (i, r) in r.iter().enumerate() {
        out += l[i] * r;
    }
    out
}

struct Transformer {
    embedding: Embedding,
    layers: [ResBlock; N_LAYERS],
    unembedding: Unembedding,
}

struct Embedding([State; N_VOCAB]);

// Just a method you can call on "Embedding"
impl Embedding {
    fn apply(&self, tok: Token) -> State {
        self.0[tok as usize].clone()
    }
}

struct LogitFn(Query);
impl LogitFn {
    fn apply(&self, st: &State) -> f32 {
        self.0.query(st)
    }
}

struct Unembedding([LogitFn; N_VOCAB]);

impl Unembedding {
    fn apply(&self, state: &State) -> Logits {
        let mut out: Logits = [0.0; N_VOCAB];

        for (i, f) in self.0.iter().enumerate() {
            out[i] = f.apply(state);
        }
        out
    }
}

// Each vocabulary element has a LogitFn which converts the state into 
// a single floating-point value by querying the state according to some particular query. 

struct ResBlock {
    attn: AttnLayer,
    mlps: MLPLayer,
}

// ATTENTION LAYER 

struct AttnLayer {
    heads: [AttnHead; N_HEADS],
}

type AttnVector = [f32; D_HEAD];
// This is the output of the attention layer 

// Mult
struct AttnHead {
    W_Q: Box<dyn Fn(&State) -> AttnVector>, 
    W_K: Box<dyn Fn(&State) -> AttnVector>,
    W_V: Box<dyn Fn(&State) -> AttnVector>,
    W_O: Box<dyn Fn(&AttnVector) -> Update>,
}

// Attention Head Implementation 

impl AttnHead {
    fn apply(&self, states: &[State]) -> Vec<Update> {
        // Apply the Q, K, and V projections to produce Q, K, and V 

        let qs: Vec<AttnVector> = states.iter().map(&self.W_Q).collect();
        let ks: Vec<AttnVector> = states.iter().map(&self.W_K).collect(); 
        let vs: Vec<AttnVector> = states.iter().map(&self.W_V).collect();


        // Iterate over each token position to compute the output at that position
        let mut values: Vec<_> = states.iter().map(|_| [0.0; D_HEAD]).collect();

        for(src, my_q) in qs.iter().enumerate() {
            let mut scores = Vec::with_capacity(src);

            // We can't get ahead! We can only look at the index we are at. 
            let visible_indices = 0..=src;
            for i in visible_indices.clone() {
                scores.push(dot(my_q, &ks[i]));

                // Dotting that q vector with the keys, then pushing them into scores 
            }

            // Turn scores into probability distribution
            softmax(&mut scores);

            // Loop over each visible position, weight their V vector by their attention weight and sum them together
            for i in visible_indices {
                let score = scores[i];
                let v = vs[i];
                for (j, vj) in v.iter().enumerate() {
                    values[src][j] += vj * score;
                }
            }
        }

        values.iter().map(&self.W_O).collect() 
    }
}

// Attention Layer: Applies each attention and sums outputs 

impl AttnLayer {
    fn apply(&self, states: &[State]) -> Vec<Update> {
        let mut updates: Vec<Update> = states.iter().map(|_| State::zero()).collect();

        for h in self.heads.iter() {
            let head_out = h.apply(states);

            updates = updates
                .iter()
                .zip(head_out.iter())
                .map(|(l, r)| l.update(r))
                .collect();
        }

        updates
    }
}

struct Neuron {
     read: Query, 
     write: Update, 
}

struct MLPLayer {
    mlps: [Neuron; D_MLP], 
    nonlinear: fn(f32) -> f32, 
}

impl MLPLayer {
    fn apply(&self, state: &State) -> Update {
        let mut out: Update = State::zero();
        for mlp in self.mlps.iter() {
            let pre_act = mlp.read.query(state);
            let post_act = (self.nonlinear)(pre_act);
            let unit_out: Update = State(mlp.write.0.map(|f| f * post_act));
            out = out.update(&unit_out)
        }
        out 
    }

}

// Now here is the full Transformer Model! 

impl ARModel for Transformer {
    fn apply(&self, tokens: &[Token]) -> Vec <Logits> {
        // Embeddings: Convert tokens to initial states 
        let mut states = tokens 
            .iter() 
            .map(|t| self.embedding.apply(*t))
            .collect::<Vec<_>>();

        // At this point we have all the initial states after embedding 

        // Pass the initial hidden state through each layer 
        // This applies the operations of all the attention layers 

        for layer in self.layers.iter(){
            let attn_out = layer.attn.apply(&states);
            states = states 
                .iter() 
                .zip(attn_out.iter())
                .map(|(l, r)| l.update(r))
                .collect();

            for i in 0..states.len() {
                // Apply the mlps to each state 
                let mlp_out = layer.mlps.apply(&states[i]);
                states[i] = states[i].update(&mlp_out);
            }
        }

        // Apply the unembedding to get out the logits 
        states.iter().map(|s| self.unembedding.apply(s)).collect()
    }
}